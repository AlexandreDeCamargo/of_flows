"""
Discrete normalizing flows via composition of radial layers.

Two families are implemented:

1. DiscreteRadialFlow  (equivariant, multi-center)
       x' = x + Σ_I α_I(|x - R_I|) * (x - R_I)
   α_I are scalar MLPs conditioned on |x-R_I| and the atomic species.
   Physically motivated for molecules: nuclear positions are the centers.

2. RezendeRadialFlow  (single-center per layer)
       x' = x + β_k h(α_k, r_k)(x - z₀_k),   r_k = |x - z₀_k|,  h = 1/(α+r)
   Closed-form log|det J| = 2 log|1+βh| + log|1+βαh²|  (no 3×3 det needed).
   Reference: Rezende & Mohamed, "Variational Inference with Normalizing Flows",
   ICML 2015.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array


class _ScalarMLP(eqx.Module):
    """
    Maps (r_norm: scalar, z_one_hot: (n_species,)) → scalar coefficient α.

    Used to compute per-atom radial coefficients in each flow layer.
    """
    linear_in: eqx.nn.Linear
    blocks: list
    linear_out: eqx.nn.Linear

    def __init__(self, n_in: int, dim: int, key):
        keys = jax.random.split(key, 4)
        self.linear_in  = eqx.nn.Linear(n_in, dim, key=keys[0])
        self.blocks      = [eqx.nn.Linear(dim, dim, key=k)
                            for k in jax.random.split(keys[1], 3)]
        self.linear_out  = eqx.nn.Linear(dim, 1, key=keys[2])

    def __call__(self, r: Array, z_one_hot: Array) -> Array:
        x = jnp.concatenate([r.reshape(1,), z_one_hot])
        x = jnp.tanh(self.linear_in(x))
        for block in self.blocks:
            x = jnp.tanh(block(x))
        return self.linear_out(x).squeeze()  # scalar


class RadialLayer(eqx.Module):
    """
    One equivariant radial layer:

        T(x) = x + Σ_I α_I(|x - R_I|) * (x - R_I)

    The Jacobian ∂T/∂x is a 3×3 matrix computed via jax.jacrev,
    so the log-det and score update are exact.
    """
    xyz_nuclei: Array   # (n_atoms, 3)  – fixed nuclear positions
    z_one_hot:  Array   # (n_atoms, n_species)
    mlp: _ScalarMLP

    def __init__(self, dim: int, mu: Array, z_one_hot: Array, key):
        n_species       = z_one_hot.shape[-1]
        self.xyz_nuclei = mu
        self.z_one_hot  = z_one_hot
        self.mlp        = _ScalarMLP(n_in=1 + n_species, dim=dim, key=key)

    def __call__(self, x: Array) -> Array:
        """Transform a single point x: (3,) → (3,)."""
        diffs   = x[None, :] - self.xyz_nuclei          # (n_atoms, 3)
        r_norms = jnp.linalg.norm(diffs, axis=-1)       # (n_atoms,)
        scalars = jax.vmap(self.mlp)(r_norms, self.z_one_hot)  # (n_atoms,)
        return x + jnp.einsum('ij,i->j', diffs, scalars)


class RezendeRadialLayer(eqx.Module):
    """
    Single-center radial flow layer (Rezende & Mohamed, ICML 2015).

        f(z) = z + β h(α, r)(z - z₀)

    where  r = |z - z₀|,  h(α, r) = 1/(α + r).

    Constraints (positive determinant everywhere):
        α = softplus(α_raw)  > 0
        β = softplus(β_raw) - α  ∈ (-α, ∞)

    At initialisation  α_raw = β_raw = 0  →  α = log2,  β = 0  (identity map).

    Exact closed-form log|det J|:
        log|det J| = 2 log|1 + βh| + log|1 + βαh²|
    which is O(1) — no 3×3 Jacobian computation needed.
    """
    z0:       Array   # (3,)  learnable reference point
    alpha_raw: Array  # scalar
    beta_raw:  Array  # scalar
    eps: float = 1e-8

    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        # initialise z₀ near the origin with small noise
        self.z0        = jax.random.normal(k1, (3,)) * 0.01
        self.alpha_raw = jnp.zeros(())
        self.beta_raw  = jnp.zeros(())

    # ── constrained parameters ────────────────────────────────────────────────
    @property
    def alpha(self) -> Array:
        return jax.nn.softplus(self.alpha_raw)           # > 0

    @property
    def beta(self) -> Array:
        return jax.nn.softplus(self.beta_raw) - self.alpha  # ∈ (-α, ∞)

    # ── forward map ───────────────────────────────────────────────────────────
    def __call__(self, z: Array) -> Array:
        """Transform a single point z: (3,) → (3,)."""
        u = z - self.z0
        r = jnp.sqrt(jnp.sum(u ** 2) + self.eps ** 2)
        h = 1.0 / (self.alpha + r)
        return z + self.beta * h * u

    # ── exact log|det J| ──────────────────────────────────────────────────────
    def log_det_jacobian(self, z: Array) -> Array:
        """Scalar log|det J(z)| using the closed-form formula."""
        u = z - self.z0
        r = jnp.sqrt(jnp.sum(u ** 2) + self.eps ** 2)
        h = 1.0 / (self.alpha + r)
        return (2.0 * jnp.log(jnp.abs(1.0 + self.beta * h))
                + jnp.log(jnp.abs(1.0 + self.beta * self.alpha * h ** 2)))

    # ── analytic inverse ──────────────────────────────────────────────────────
    def inverse(self, y: Array) -> Array:
        """Invert a single point y: (3,) → (3,).

        Solves  y = x + β h(r)(x − z₀)  exactly via the quadratic identity.

        Let  v = y − z₀,  d = |v|.  Since the flow is radial the input x lies
        on the ray  z₀ + r·v̂  (same direction as y − z₀), so we only need the
        scalar r = |x − z₀|.  Substituting into the forward map gives:

            d = r (α + r + β) / (α + r)
            ⟹  r² + (α + β − d) r − d α = 0

        Positive root (discriminant is always ≥ 0 under the β > −α constraint):

            r_inv = [ (d − α − β) + √((α + β − d)² + 4 d α) ] / 2
        """
        v = y - self.z0
        d = jnp.sqrt(jnp.sum(v ** 2) + self.eps ** 2)
        b_coeff     = self.alpha + self.beta - d           # (α+β−d)
        discriminant = b_coeff ** 2 + 4.0 * d * self.alpha
        r_inv = (-b_coeff + jnp.sqrt(jnp.maximum(discriminant, 0.0))) / 2.0
        return self.z0 + r_inv * v / d


class RezendeRadialFlow(eqx.Module):
    """
    Composition of K single-center radial layers (Rezende & Mohamed 2015).

    Each layer has its own learnable centre z₀_k, α_k, β_k.
    Compatible with forward_drf (uses model.layers) but also supports
    forward_rdm which exploits the closed-form log-det for efficiency.
    """
    layers: list

    def __init__(self, n_layers: int, key):
        keys = jax.random.split(key, n_layers)
        self.layers = [RezendeRadialLayer(k) for k in keys]

    def __call__(self, z: Array) -> Array:
        for layer in self.layers:
            z = layer(z)
        return z


class DiscreteRadialFlow(eqx.Module):
    """
    Discrete equivariant normalizing flow: composition of K RadialLayer
    transformations.

    Parameters
    ----------
    n_layers : int
        Number of radial layers K.
    dim : int
        Hidden dimension of each scalar MLP.
    mu : (n_atoms, 3) array
        Nuclear coordinates (reference points for equivariance).
    z_one_hot : (n_atoms, n_species) array
        One-hot encoding of atomic numbers.
    key : PRNGKey
    """
    layers: list

    def __init__(self, n_layers: int, dim: int, mu: Array,
                 z_one_hot: Array, key):
        keys = jax.random.split(key, n_layers)
        self.layers = [RadialLayer(dim, mu, z_one_hot, k) for k in keys]

    def __call__(self, x: Array) -> Array:
        """Forward pass: transform a single point x: (3,) → (3,)."""
        for layer in self.layers:
            x = layer(x)
        return x
