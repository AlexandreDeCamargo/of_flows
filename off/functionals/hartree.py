import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class CoulombPotential_(eqx.Module):
    """Hartree repulsion, element-wise pairs: ½Ne²/|x - x'|."""

    eps: float

    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, inp) -> Float[Array, "batch 1"]:
        x, xp, Ne = inp.x, inp.xp, inp.Ne
        z = jnp.sum((x - xp) ** 2 + self.eps, axis=-1, keepdims=True)
        return 0.5 * (Ne ** 2) / jnp.sqrt(z)


class CoulombPotential(eqx.Module):
    """Hartree repulsion averaged over all batch² pairs (lower variance)."""

    eps: float

    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, inp) -> Float[Array, "batch 1"]:
        x, xp, Ne = inp.x, inp.xp, inp.Ne
        x2 = jnp.sum(x * x, axis=-1)
        xp2 = jnp.sum(xp * xp, axis=-1)
        r2 = x2[:, None] + xp2[None, :] - 2.0 * (x @ xp.T)
        r2 = jnp.maximum(r2, 0.0) + self.eps
        v = jnp.mean(1.0 / jnp.sqrt(r2), axis=-1, keepdims=True)
        return 0.5 * (Ne ** 2) * v
