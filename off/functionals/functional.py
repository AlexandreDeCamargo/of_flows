import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Any, Callable, NamedTuple, Optional


class FunctionalInputs(NamedTuple):
    """Inputs passed to every functional; each uses only the fields it needs."""
    den:   Any
    score: Any
    x:     Any
    Ne:    Any
    mol:   Any
    xp:    Any = None


def unit_coefficient(self, *_):
    """Constant weight 1 — for fixed (non-learned) functionals."""
    return jnp.array([[1.0]])


class Functional(eqx.Module):

    coefficients:       Callable
    energy_densities:   Callable
    coefficient_inputs: Optional[Callable] = None

    def __call__(self, inp: FunctionalInputs) -> Float[Array, "batch 1"]:
        e  = self.energy_densities(inp)
        ci = self.coefficient_inputs(inp) if self.coefficient_inputs is not None else None
        c  = self.coefficients(self, ci)
        return jnp.sum(c * e, axis=-1, keepdims=True)


class CompositeFunctional(eqx.Module):
    """Sum of several functionals."""

    functionals: list

    def __init__(self, *functionals):
        self.functionals = functionals

    def __call__(self, inp: FunctionalInputs) -> Float[Array, "batch 1"]:
        result = 0.0
        for func in self.functionals:
            result = result + func(inp)
        return result

    def __add__(self, other):
        if isinstance(other, CompositeFunctional):
            return CompositeFunctional(self.functionals + other.functionals)
        return CompositeFunctional(self.functionals + [other])


class EnergyFunctional(eqx.Module):
    """Total energy: sums the component functionals on a shared FunctionalInputs."""

    kinetic:         Any
    external:        Any
    hartree:         Any
    exchange:        Any
    correlation:     Any = None
    core_correction: Any = None

    @staticmethod
    def _integrate(energy_density, weights):
        return jnp.sum(weights * energy_density)

    def terms(self, inp: FunctionalInputs) -> dict:
        return {
            "kin":  self.kinetic(inp),
            "vnuc": self.external(inp),
            "hart": self.hartree(inp),
            "x":    self.exchange(inp),
            "c":    self.correlation(inp) if self.correlation is not None else 0.0,
            "cc":   self.core_correction(inp) if self.core_correction is not None else 0.0,
        }

    def __call__(self, inp: FunctionalInputs):
        t = self.terms(inp)
        return t["kin"] + t["vnuc"] + t["hart"] + t["x"] + t["c"] + t["cc"]
