import jax.numpy as jnp
from jax import lax
from .functional import Functional, CompositeFunctional, unit_coefficient

C_TF = (3. / 10.) * (3. * jnp.pi ** 2) ** (2 / 3)


def tf_eps(den, score, x, Ne, mol, xp, c=C_TF):
    r"""
    Thomas-Fermi kinetic functional.

    See paper eq. 2 in https://pubs.aip.org/aip/jcp/article/114/2/631/184186/Thomas-Fermi-Dirac-von-Weizsacker-models-in-finite

    T_{\text{TF}}[\rho] = c \int \rho(\boldsymbol{x})^{5/3} d\boldsymbol{x}
                        = c \int \rho(\boldsymbol{x})^{2/3} \rho(\boldsymbol{x}) d\boldsymbol{x}
    T_{\text{TF}}[\rho] = c\, N_e^{5/3}\, \mathbb{E}_{\rho_\phi}\left[ \rho_\phi(\boldsymbol{x})^{2/3} \right]

    with c = \frac{3}{10}(3\pi^2)^{2/3}.

    Parameters
    ----------
    den : Array
        Density.
    Ne : int
        Number of electrons.
    c : float, optional
        Prefactor, by default (3/10)(3 pi^2)^(2/3).

    Notes
    -----
    score, x, mol, xp are accepted for the shared functional interface but unused here.

    Returns
    -------
    jax.Array
        Thomas-Fermi kinetic energy.
    """
    return c * (Ne ** (5 / 3)) * den ** (2 / 3)


def weizsacker_eps(den, score, x, Ne, mol, xp, lam=0.2):
    r"""
    von Weizsacker gradient correction.

    See paper eq. 3 in https://pubs.aip.org/aip/jcp/article/114/2/631/184186/Thomas-Fermi-Dirac-von-Weizsacker-models-in-finite

    T_{\text{Weizsacker}}[\rho] = \frac{\lambda}{8} \int \frac{(\nabla \rho)^2}{\rho} d\boldsymbol{x}
                                = \frac{\lambda}{8} \int \rho \left(\frac{\nabla \rho}{\rho}\right)^2 d\boldsymbol{x}
    T_{\text{Weizsacker}}[\rho] = \frac{\lambda N_e}{8}\, \mathbb{E}_{\rho_\phi}\left[ \left(\frac{\nabla \rho}{\rho}\right)^2 \right]

    Parameters
    ----------
    score : Array
        Gradient of the log-density, s = (nabla rho)/rho.
    Ne : int
        Number of electrons.
    lam : float, optional (W. Stich, E.K.U. Gross, Z. Physik A 309(1):511, 1982)
        Phenomenological parameter lambda, by default 0.2.

    Notes
    -----
    den, x, mol, xp are accepted for the shared functional interface but unused here.

    Returns
    -------
    jax.Array
        von Weizsacker kinetic energy.
    """
    score_sqr = jnp.einsum('ij,ij->i', score, score)
    return (lam * Ne / 8.) * lax.expand_dims(score_sqr, (1,))


tf = Functional(coefficients=unit_coefficient, energy_per_particle=tf_eps)

def weizsacker(lam=0.2):
    r"""von Weizsacker functional with prefactor ``lam`` (see :func:`weizsacker_eps`)."""
    return Functional(
        coefficients=unit_coefficient,
        energy_per_particle=lambda den, score, x, Ne, mol, xp: weizsacker_eps(
            den, score, x, Ne, mol, xp, lam),
    )


def tf_weizsacker(lam=0.2):
    r"""TF-lambda-W kinetic functional:  T = T_TF + lambda * T_W."""
    return CompositeFunctional(tf, weizsacker(lam))
