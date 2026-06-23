import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from jax import vmap


class NuclearPotential(eqx.Module):
    """Electron-nuclei attraction: -Ne·Σ_i Z_i/|x - R_i|."""

    eps: float

    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, inp) -> Float[Array, "batch"]:
        x, Ne, mol = inp.x, inp.Ne, inp.mol

        def _potential(x, molecule):
            r = jnp.sqrt(jnp.sum((x - molecule['coords']) ** 2, axis=1)) + self.eps
            return molecule['z'] / r

        r = vmap(_potential, in_axes=(None, 0), out_axes=-1)(x, mol)
        return -Ne * jnp.sum(r, axis=-1, keepdims=True)
