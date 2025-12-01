import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Scalar
from jax import vmap,lax 

class KatoCondition(eqx.Module):
    r"""
    Kato cusp condition functional.
    
    Ensures correct electron-nucleus cusp behavior. 
    
    Attributes
    ----------
    a : Float[Array, ""]
        Prefactor constant, default is 2.0/3.0. 
    dim : Int[Scalar, ""]
        Dimension of the system, default is 3 dimensions.
    eps : Float[Array, ""], 
        Float for numerical stability. 
    """
    
    a: Float[Array, ""]
    dim: Int[Scalar, ""]
    eps: Float[Array, ""]
    
    def __init__(self, a = 2.0/3.0, eps = 1e-5, dim = 3):
        self.a = a 
        self.eps = eps
        self.dim = dim 

    def __call__(
        self,
        x: Float[Array, "batch dim"],
        den: Float[Array, "batch"],
        score: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
        molecule: Array,
    ) -> Float[Array, "batch"]:
        r"""
        Compute Kato condition term.
        
        Parameters
        ----------
        x : Float[Array, "batch d"]
            Spatial points.
        den : Float[Array, "batch"]
            Electron density at each batch point.
        score : Float[Array, "batch d"]
            Gradient of log-likelihood (score function).
        Ne : Int[Scalar, ""]
            Number of electrons.
        molecule : Array
            Molecular information containing coordinates and charges.
            
        Returns
        -------
        Float[Array, "batch"]
            Kato condition term at each batch point.
        """
        
        def _wi(x: Array, molecule: Array) -> Array:
            r = jnp.sqrt(
                jnp.sum((x - molecule['coords']) * (x - molecule['coords']), axis=1)
            ) 
            z = molecule['z']
            w_i = jnp.exp(-self.a * z * r)
            return w_i

        _wi_mapped = vmap(_wi, in_axes=(None, 0), out_axes=-1)(x, molecule)
        
        score_sqr = jnp.einsum('ij,ij->i', score, score)
        weizs = (0.2 / 8.0) * lax.expand_dims(score_sqr, (1,))
        # weizs = (0.2 / 8.0) * lax.expand_dims(score_sqr, (1,))
        
        kinetic = jnp.abs(weizs - (molecule['z']**2 / 2))
        term = jnp.sum(kinetic * _wi_mapped, axis=-1, keepdims=True)
        
        return Ne * term
    