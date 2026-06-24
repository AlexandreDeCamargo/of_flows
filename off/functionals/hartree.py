import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class CoulombPotential_(eqx.Module):
    r"""
    Classical electron-electron repulsion (Hartree) potential, element-wise pairs.

    Pairs each x_i with x'_i (the two half-batches), giving `batch` Monte-Carlo
    pairs:

    V_{\text{Hartree}}(\boldsymbol{x}, \boldsymbol{x}') = \frac{1}{2} N_e^2 \frac{1}{|\boldsymbol{x} - \boldsymbol{x}'|}

    Parameters
    ----------
    eps : float, optional
        Small constant for numerical stability, by default 1e-5.
    """

    eps: float

    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, den, score, x, Ne, mol, xp) -> Float[Array, "batch 1"]:
        r"""Uses x, xp (paired points) and Ne; den, score, mol unused. Returns ½Ne²/|x-x'| per pair."""
        z = jnp.sum((x - xp) ** 2 + self.eps, axis=-1, keepdims=True)
        return 0.5 * (Ne ** 2) / jnp.sqrt(z)


class CoulombPotential(eqx.Module):
    r"""
    Classical electron-electron repulsion (Hartree) potential, all-pairs estimator.

    Same physics as :class:`CoulombPotential_` (true 1/|x-x'| Coulomb), but averages
    each x_i against *every* x'_j (the full batch x batch double sum), i.e. batch^2
    pairs instead of batch. This is the same estimator the grid quadrature uses and
    has substantially lower Monte-Carlo variance for the same samples, at the cost
    of a (batch, batch) distance matrix.

    V_{\text{Hartree}} = \frac{1}{2} N_e^2 \left\langle \frac{1}{|\boldsymbol{x}_i - \boldsymbol{x}'_j|} \right\rangle_j

    Parameters
    ----------
    eps : float, optional
        Small constant for numerical stability, by default 1e-5.
    """

    eps: float

    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, den, score, x, Ne, mol, xp) -> Float[Array, "batch 1"]:
        r"""Uses x, xp and Ne; den, score, mol unused. Returns the per-point Hartree potential."""
        x2 = jnp.sum(x * x, axis=-1)
        xp2 = jnp.sum(xp * xp, axis=-1)
        r2 = x2[:, None] + xp2[None, :] - 2.0 * (x @ xp.T)
        r2 = jnp.maximum(r2, 0.0) + self.eps
        v = jnp.mean(1.0 / jnp.sqrt(r2), axis=-1, keepdims=True)
        return 0.5 * (Ne ** 2) * v
