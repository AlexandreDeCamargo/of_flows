import jax.numpy as jnp
from jax import lax
from functionals.functional import Functional, CompositeFunctional, unit_coefficient


def lda_density(inp):
    """Dirac/LDA exchange: -¾(3/π)^(1/3)·Ne^(4/3)·ρ^(1/3)."""
    l = -(3 / 4) * (inp.Ne ** (4 / 3)) * (3 / jnp.pi) ** (1 / 3)
    return l * inp.den ** (1 / 3)


def b88_density(inp, clip_cte=1e-30, beta=0.0042):
    """B88 exchange."""
    den, score, Ne = inp.den, inp.score, inp.Ne
    den_clipped = jnp.clip(den, clip_cte)
    log_den = jnp.log2(den_clipped)
    score_sqr = jnp.einsum('ij,ij->i', score, score)
    grad_den_norm_sq = lax.expand_dims(score_sqr, (1,)) * den_clipped * den_clipped
    log_grad_den_norm = jnp.log2(jnp.clip(grad_den_norm_sq, clip_cte)) / 2
    log_x_sigma = log_grad_den_norm - 4 / 3.0 * log_den
    x_sigma = 2 ** log_x_sigma
    b88_e = -(beta * 2 ** (4 * log_den / 3 + 2 * log_x_sigma
                           - jnp.log2(1 + 6 * beta * x_sigma * jnp.arcsinh(x_sigma))))
    return b88_e * Ne ** (2 / 3)


def vwn_density(inp, clip_cte=1e-30):
    """VWN correlation."""
    den, Ne = inp.den, inp.Ne
    A, b, c, x0 = 0.0621814, 3.72744, 12.9352, -0.10498
    den_clipped = jnp.where(den > clip_cte, den, 0.0)
    log_den = jnp.log2(jnp.clip(den_clipped, clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_den / 3.0
    log_x = log_rs / 2
    x = 2. ** log_x
    X = 2. ** (2. * log_x) + 2. ** (log_x + jnp.log2(b)) + c
    X0 = x0 ** 2 + b * x0 + c
    Q = jnp.sqrt(4 * c - b ** 2)
    e_PF = A / 2. * (
        2. * jnp.log(x) - jnp.log(X) + 2. * b / Q * jnp.arctan(Q / (2. * x + b))
        - b * x0 / X0 * (jnp.log((x - x0) ** 2. / X)
                         + 2. * (2. * x0 + b) / Q * jnp.arctan(Q / (2. * x + b)))
    )
    return Ne * e_PF


def pw92_density(inp, clip_cte=1e-30):
    """PW92 correlation."""
    den, Ne = inp.den, inp.Ne
    A_, alpha1 = 0.031091, 0.21370
    beta1, beta2, beta3, beta4 = 7.5957, 3.5876, 1.6382, 0.49294
    log_den = jnp.log2(jnp.clip(den, clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_den / 3.0
    brs_1_2 = 2 ** (log_rs / 2 + jnp.log2(beta1))
    ars = 2 ** (log_rs + jnp.log2(alpha1))
    brs = 2 ** (log_rs + jnp.log2(beta2))
    brs_3_2 = 2 ** (3 * log_rs / 2 + jnp.log2(beta3))
    brs2 = 2 ** (2 * log_rs + jnp.log2(beta4))
    e_PF = -2 * A_ * (1 + ars) * jnp.log(1 + (1 / (2 * A_)) / (brs_1_2 + brs + brs_3_2 + brs2))
    return Ne * e_PF


lda  = Functional(coefficients=unit_coefficient, energy_densities=lda_density)
b88  = Functional(coefficients=unit_coefficient, energy_densities=b88_density)
vwn  = Functional(coefficients=unit_coefficient, energy_densities=vwn_density)
pw92 = Functional(coefficients=unit_coefficient, energy_densities=pw92_density)


def lda_b88():
    return CompositeFunctional(lda, b88)
