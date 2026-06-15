"""
Promolecular STO-3G electron density as a Distrax distribution.

Each atom contributes |psi|^2 (an exact Gaussian mixture built from its STO-3G
contraction); the molecular density is the electron-count-weighted sum of the
atomic densities, normalized to a probability distribution (integral = 1) to
match the convention used by ProMolecularDensity.

Interface required by utils.batch_generator + rho_on_grid:
    sample(seed=key, sample_shape=N)  -> (N, 3)
    log_prob(x)                       -> (N, 1)        (2-D, not (N,)!)
    score(x)                          -> (N, 3)        ( = grad_x log p )

Scope: hydrogen (1s) only -> use for H, H2, H10, ... (all-hydrogen systems).
Heavier atoms need their full STO-3G shell structure (2s, 2p, occupations),
which is NOT handled here.

Coordinates in bohr.  STO-3G data from Basis Set Exchange (basissetexchange.org).
"""

import jax
import jax.numpy as jnp
import jax.random as jrnd
import distrax

# STO-3G s-shell parameters per element Z: (exponents, raw contraction coeffs).
# H from BSE (STO-3G, hehre1969a).  Add more elements' s shells here as needed.
STO3G_S = {
    1: (
        [0.3425250914e1, 0.6239137298e0, 0.1688554040e0],
        [0.1543289673e0, 0.5353281423e0, 0.4446345422e0],
    ),
}


def _atom_mixture_components(exps, coeffs_raw, dtype):
    """Return (weights(9,), sigmas(9,)) for one normalized 1s STO-3G atom.

    rho_atom(r) = |psi|^2 = sum_{k,l} A_kl exp(-(a_k+a_l) r^2),  integral = 1,
    with A_kl = (c_k N_k)(c_l N_l).  Each exp(-beta r^2) is an isotropic
    Gaussian N(0, sigma^2 I) with sigma^2 = 1/(2 beta) and mixing weight
    A_kl (pi/beta)^{3/2}.
    """
    a     = jnp.asarray(exps,       dtype=dtype)          # (3,)
    c_raw = jnp.asarray(coeffs_raw, dtype=dtype)          # (3,)
    N     = (2.0 * a / jnp.pi) ** 0.75                    # primitive norms (3,)

    # Normalize the contraction so that <psi|psi> = 1.
    S = N[:, None] * N[None, :] * (jnp.pi / (a[:, None] + a[None, :])) ** 1.5
    c = c_raw / jnp.sqrt(c_raw @ S @ c_raw)               # (3,)

    amp   = c * N                                         # (3,)
    beta  = a[:, None] + a[None, :]                       # (3,3)
    A     = amp[:, None] * amp[None, :]                   # (3,3)
    w     = (A * (jnp.pi / beta) ** 1.5).reshape(-1)      # (9,)  sums to 1
    sigma = jnp.sqrt(1.0 / (2.0 * beta)).reshape(-1)      # (9,)
    return w, sigma


class ProMolecularSTO3G(distrax.Distribution):
    """STO-3G promolecular density (H only) as a sample-able Distrax mixture.

    Constructed exactly like ProMolecularDensity:  ProMolecularSTO3G(z, coords).
    """

    def __init__(self, z, coords, dtype=jnp.float64):
        self._dtype = dtype
        z      = jnp.asarray(z, dtype=dtype).ravel()      # (n_atoms,)
        coords = jnp.asarray(coords, dtype=dtype)         # (n_atoms, 3)

        z_int = [int(round(float(zi))) for zi in z]
        if any(zi != 1 for zi in z_int):
            raise NotImplementedError(
                "ProMolecularSTO3G currently supports hydrogen (1s) only; "
                f"got Z={z_int}.  Add the full shell structure for heavier atoms.")

        Ne = float(jnp.sum(z))                            # total electrons
        locs, scales, weights = [], [], []
        for zi, Ri in zip(z_int, coords):
            exps, coeffs = STO3G_S[zi]
            w, sigma = _atom_mixture_components(exps, coeffs, dtype)   # (9,),(9,)
            n_el = float(zi)                              # electrons on this atom (H: 1)
            weights.append((n_el / Ne) * w)              # this atom's share of the pdf
            locs.append(jnp.broadcast_to(Ri, (w.shape[0], 3)))
            scales.append(sigma[:, None] * jnp.ones((w.shape[0], 3), dtype))

        self.weights = jnp.concatenate(weights)          # (9*n_atoms,)  sums to 1
        self.locs    = jnp.concatenate(locs)             # (9*n_atoms, 3)
        self.scales  = jnp.concatenate(scales)           # (9*n_atoms, 3)
        self.Ne      = Ne

        self._dist = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(probs=self.weights),
            components_distribution=distrax.MultivariateNormalDiag(
                loc=self.locs, scale_diag=self.scales),
        )

    # ── distrax plumbing ──────────────────────────────────────────────────────
    @property
    def event_shape(self):
        return (3,)

    @property
    def batch_shape(self):
        return ()

    @property
    def dtype(self):
        return self._dtype

    def _sample_n(self, key, n):
        return self._dist.sample(seed=key, sample_shape=(n,))

    # ── interface required by utils.batch_generator / rho_on_grid ─────────────
    def sample(self, *, seed, sample_shape=()):
        return self._dist.sample(seed=seed, sample_shape=sample_shape)

    def log_prob(self, x):
        """log p(x); shape (..., 1) to match the prior contract."""
        return self._dist.log_prob(x)[..., None]

    def prob(self, x):
        return jnp.exp(self._dist.log_prob(x))[..., None]

    def score(self, values):
        """grad_x log p(x); shape (N, 3)."""
        return jax.vmap(jax.grad(lambda y: self._dist.log_prob(y)))(values)

    def sample_and_log_prob(self, *, seed, sample_shape=()):
        x = self._dist.sample(seed=seed, sample_shape=sample_shape)
        return x, self.log_prob(x)


if __name__ == "__main__":
    # H2 self-test at R = 1.4 bohr.
    jax.config.update("jax_enable_x64", True)
    key = jrnd.PRNGKey(0)
    R = 1.4
    z = jnp.array([1.0, 1.0])
    coords = jnp.array([[0.0, 0.0, -R / 2], [0.0, 0.0, R / 2]])

    rho = ProMolecularSTO3G(z, coords)
    x  = rho.sample(seed=key, sample_shape=20000)
    lp = rho.log_prob(x)
    sc = rho.score(x)

    print("x   shape   :", x.shape)                       # (20000, 3)
    print("lp  shape   :", lp.shape)                      # (20000, 1)
    print("sc  shape   :", sc.shape)                      # (20000, 3)
    print("weights sum :", float(jnp.sum(rho.weights)))   # 1.0
    print("mean pos    :", jnp.mean(x, axis=0))           # ~[0,0,0] by symmetry
    print("mean |z|    :", float(jnp.mean(jnp.abs(x[:, 2]))))
    print("frac z<0    :", float(jnp.mean(x[:, 2] < 0)))  # ~0.5 (two centers)

    # correctness: exp(log_prob) must equal (1/Ne) * sum_atoms |psi(x-R)|^2
    def ref(xq):
        a, c_raw = STO3G_S[1]
        a = jnp.asarray(a); c_raw = jnp.asarray(c_raw)
        N = (2.0 * a / jnp.pi) ** 0.75
        S = N[:, None] * N[None, :] * (jnp.pi / (a[:, None] + a[None, :])) ** 1.5
        c = c_raw / jnp.sqrt(c_raw @ S @ c_raw)
        amp = c * N

        def psi2(d):
            r2 = jnp.sum(d ** 2, axis=-1, keepdims=True)
            g  = amp * jnp.exp(-a * r2)
            return jnp.sum(g, axis=-1) ** 2

        return sum(psi2(xq - Ri) for Ri in coords) / 2.0

    err = float(jnp.max(jnp.abs(jnp.exp(lp).ravel() - ref(x))))
    print("max|exp(logp)-ref|:", err)                     # ~1e-12 if correct
