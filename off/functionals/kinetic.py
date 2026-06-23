import jax.numpy as jnp
from jax import lax
from functionals.functional import Functional, CompositeFunctional, unit_coefficient

C_TF = (3. / 10.) * (3. * jnp.pi ** 2) ** (2 / 3)


def tf_density(inp, c=C_TF):
    """Thomas-Fermi: c·Ne^(5/3)·ρ^(2/3)."""
    return c * (inp.Ne ** (5 / 3)) * inp.den ** (2 / 3)


def weizsacker_density(inp, lam=0.2):
    """von Weizsäcker: (λ·Ne/8)·|∇log ρ|²."""
    score_sqr = jnp.einsum('ij,ij->i', inp.score, inp.score)
    return (lam * inp.Ne / 8.) * lax.expand_dims(score_sqr, (1,))


tf = Functional(coefficients=unit_coefficient,
                energy_densities=tf_density)


def weizsacker(lam=0.2):
    return Functional(coefficients=unit_coefficient,
                      energy_densities=lambda inp: weizsacker_density(inp, lam))


def tf_weizsacker(lam=0.2):
    """T_TF + λ·T_W."""
    return CompositeFunctional(tf, weizsacker(lam))
