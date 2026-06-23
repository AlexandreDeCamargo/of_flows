import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from jax import vmap, lax


class KatoCondition(eqx.Module):
    """Kato cusp-condition penalty."""

    a: float
    eps: float

    def __init__(self, a=2.0/3.0, eps=1e-5):
        self.a = a
        self.eps = eps

    def __call__(self, inp) -> Float[Array, "batch"]:
        x, score, Ne, mol = inp.x, inp.score, inp.Ne, inp.mol

        def _wi(x, molecule):
            r = jnp.sqrt(jnp.sum((x - molecule['coords']) ** 2, axis=1) + self.eps ** 2)
            return jnp.exp(-self.a * molecule['z'] * r)

        wi = vmap(_wi, in_axes=(None, 0), out_axes=-1)(x, mol)
        score_sqr = jnp.einsum('ij,ij->i', score, score)
        weizs = (1.0 / 8.0) * lax.expand_dims(score_sqr, (1,))
        kinetic = jnp.abs(weizs - (mol['z'] ** 2 / 2))
        return Ne * jnp.sum(kinetic * wi, axis=-1, keepdims=True)


class HutcheonCuspCondition(eqx.Module):
    """Exact nuclear cusp cost (Hutcheon & Wibowo-Teale, PRB 110, 195146)."""

    eps: float

    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, inp) -> Float[Array, "batch 1"]:
        x, den, score, Ne, mol = inp.x, inp.den, inp.score, inp.Ne, inp.mol
        coords, z = mol['coords'], mol['z']

        def per_atom(R_i, Z_i):
            r_vec = x - R_i
            r_norm = jnp.sqrt(jnp.sum(r_vec ** 2, axis=-1) + self.eps ** 2)
            r_hat = r_vec / r_norm[:, None]
            w_i = jnp.sqrt(jnp.pi / Z_i ** 3) * jnp.exp(-Z_i * r_norm)
            cusp_sq = jnp.sum((score + 2.0 * Z_i * r_hat) ** 2, axis=-1)
            return cusp_sq * w_i

        total = jnp.sum(jax.vmap(per_atom)(coords, z), axis=0)
        return (Ne * den.squeeze(-1) * total)[:, None]
