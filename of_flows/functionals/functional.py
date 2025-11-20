import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Scalar

class Functional(eqx.Module):
    """
    Base class for all energy functionals.
    """
    
    def __call__(
        self,
        den: Float[Array, "batch"],
        score: Float[Array, "batch d"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        """
        Compute the functional value.
        
        Parameters
        ----------
        den : Float[Array, "batch"]
            Electron density at each spatial point.
        score : Float[Array, "batch d"]
            Gradient of log-likelihood (score function).
        Ne : Int[Scalar, ""]
            Number of electrons.
            
        Returns
        -------
        Float[Array, "batch"]
            Functional value at each batch point.
        """
        raise NotImplementedError

class CompositeFunctional(Functional):
    """
    A functional that is the sum of multiple functionals.
    
    Attributes
    ----------
    functionals : list
        List of Functional objects to sum together.
    """
    
    functionals: list
    
    def __init__(self, *functionals):
        self.functionals = functionals
    
    def __call__(
        self,
        den: Float[Array, "batch"],
        score: Float[Array, "batch d"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        """
        Compute the sum of all functionals.
        """
        result = 0.0
        for func in self.functionals:
            result = result + func(den, score, Ne)
        return result
    
    def __add__(self, other):
        """Allow chaining multiple additions."""
        if isinstance(other, CompositeFunctional):
            return CompositeFunctional(self.functionals + other.functionals)
        return CompositeFunctional(self.functionals + [other])