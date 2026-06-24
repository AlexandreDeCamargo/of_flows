import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Any, Callable, NamedTuple, Optional


class FunctionalInputs(NamedTuple):
    r"""
    Container holding every input a functional might need.

    It is the single bundle handed to :class:`EnergyFunctional`, which then
    *separates* it into explicit arguments for each component functional. The leaf
    functionals therefore receive plain ``(den, score, x, Ne, mol, xp)`` and never
    see this object.

    Fields
    ------
    den   : Float[Array, "batch 1"]   density shape factor rho_phi.
    score : Float[Array, "batch d"]   score, (nabla rho)/rho = nabla log rho_phi.
    x     : Float[Array, "batch d"]   sample / grid positions.
    Ne    : int                       number of electrons.
    mol   : dict                      {'coords': ..., 'z': ...} nuclear geometry / charges.
    xp    : Float[Array, "batch d"]   second set of positions for the pairwise Hartree.
    """
    den:   Any
    score: Any
    x:     Any
    Ne:    Any
    mol:   Any
    xp:    Any = None


def unit_coefficient(self, *_):
    r"""Constant unit weight (c = 1): turns a Functional into a fixed (non-learned) functional."""
    return jnp.array([[1.0]])


class Functional(eqx.Module):
    r"""
    Local density functional,  F[\rho] = \int c_\theta[\rho] \cdot e[\rho]\, d\boldsymbol{x}.

    Every functional shares the explicit signature ``(den, score, x, Ne, mol, xp)``
    -- all inputs are passed even if a given functional ignores some -- and returns
    its per-point energy density of shape (batch, 1). Assembled from two callables:

    energy_densities(den, score, x, Ne, mol, xp) -> e[rho]
        the energy densities, returned up to the rho factor.
    coefficients(self, cinputs) -> c_theta[rho]
        the weights; :func:`unit_coefficient` (c = 1) for a fixed functional, a
        network for a learned one.
    coefficient_inputs(den, score, x, Ne, mol, xp) -> features   (optional)
        features fed to ``coefficients`` (learned functionals only).
    """

    coefficients:       Callable
    energy_densities:   Callable
    coefficient_inputs: Optional[Callable] = None
    
    def __call__(self, den, score, x, Ne, mol, xp) -> Float[Array, "batch 1"]:
        e  = self.energy_densities(den, score, x, Ne, mol, xp)
        ci = (self.coefficient_inputs(den, score, x, Ne, mol, xp)
              if self.coefficient_inputs is not None else None)
        c  = self.coefficients(self, ci)
        return jnp.sum(c * e, axis=-1, keepdims=True)
    #Add a functional, or density. 

class CompositeFunctional(eqx.Module):
    r"""Sum of several functionals, all sharing the ``(den, score, x, Ne, mol, xp)`` signature."""

    functionals: list

    def __init__(self, *functionals):
        self.functionals = functionals

    def __call__(self, den, score, x, Ne, mol, xp) -> Float[Array, "batch 1"]:
        result = 0.0
        for func in self.functionals:
            result = result + func(den, score, x, Ne, mol, xp)
        return result

    def __add__(self, other):
        if isinstance(other, CompositeFunctional):
            return CompositeFunctional(self.functionals + other.functionals)
        return CompositeFunctional(self.functionals + [other])


class EnergyFunctional(eqx.Module):
    r"""
    High-level OF-DFT energy functional.

    Receives the single :class:`FunctionalInputs` bundle and *separates* it into
    explicit arguments for each component functional (kinetic, external/nuclear,
    Hartree, exchange, and optionally correlation and core-correction): every
    component is called with the same ``(den, score, x, Ne, mol, xp)`` and uses only
    what it needs. ``correlation`` and ``core_correction`` may be ``None``.

    ``terms`` returns the per-component energy densities (convenient for logging);
    ``__call__`` returns their sum. Integrate with :meth:`_integrate` (Monte-Carlo
    measure 1/N during training, or grid weights w*rho for quadrature).
    """

    kinetic:         Any
    external:        Any
    hartree:         Any
    exchange:        Any
    correlation:     Any = None
    core_correction: Any = None

    @staticmethod
    def _integrate(energy_density, weights):
        r"""Quadrature: integral ~ sum_i w_i e_i. `weights` is 1/N (Monte-Carlo) or w_i*rho_i (grid)."""
        return jnp.sum(weights * energy_density)

    def terms(self, inp: FunctionalInputs) -> dict:
        r"""Separate ``inp`` and evaluate every component on the same explicit arguments."""
        a = (inp.den, inp.score, inp.x, inp.Ne, inp.mol, inp.xp)
        return {
            "kin":  self.kinetic(*a),
            "vnuc": self.external(*a),
            "hart": self.hartree(*a),
            "x":    self.exchange(*a),
            "c":    self.correlation(*a) if self.correlation is not None else 0.0,
            "cc":   self.core_correction(*a) if self.core_correction is not None else 0.0,
        }

    def __call__(self, inp: FunctionalInputs):
        r"""Per-point total energy density (sum of all components)."""
        t = self.terms(inp)
        return t["kin"] + t["vnuc"] + t["hart"] + t["x"] + t["c"] + t["cc"]
