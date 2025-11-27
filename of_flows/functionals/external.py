import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Scalar
from jax import vmap 

class NuclearPotential(eqx.Module):
    r"""
    External electron-nuclei interaction potential.
    
    $$V_{\text{ext}}(\mathbf{x}) = -N_e \sum_i \frac{Z_i}{|\mathbf{x} - \mathbf{R}_i|}$$
    
    Attributes
    ----------
    eps : Float[Array, ""], 
        Float for numerical stability. 
    dim : Int[Scalar, ""]
        Dimension of the system, default is 3 dimensions. 
    """
    
    eps: Float[Array, ""]
    dim: Int[Scalar, ""]

    def __init__(self, eps = 1e-5, dim = 3):
        self.eps = eps
        self.dim = dim 
    
    eps: float = 1e-4
    
    def __call__(
        self,
        x: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
        mol_info: Array,
    ) -> Float[Array, "batch"]:
        r"""
        Compute nuclear potential.
        
        Parameters
        ----------
        x : Float[Array, "batch dim"]
            Points where the potential is evaluated.
        Ne : Int[Scalar, ""]
            Number of electrons.
        mol_info : Array
            Molecular information containing coordinates and charges.
            
        Returns
        -------
        Float[Array, "batch"]
            Electron-nuclei interaction potential at each point x.
        """
        
        def _potential(x: Array, molecule: Array) -> Array:
            r = jnp.sqrt(
                jnp.sum((x - molecule['coords']) * (x - molecule['coords']), axis=1)
            ) + self.eps
            z = molecule['z']
            return z / r

        r = vmap(_potential, in_axes=(None, 0), out_axes=-1)(x, mol_info)
        r = jnp.sum(r, axis=-1, keepdims=True)
        return -Ne * r


class NuclearPotential1D(eqx.Module):
    r"""
    1D attraction to nuclei of charges Z_alpha and Z_beta.
    
    See eq 7 in https://pubs.aip.org/aip/jcp/article/139/22/224104/193579/Orbital-free-bond-breaking-via-machine-learning 
    
    $$V_{\text{ext}}(x) = -N_e \left[ \frac{Z_\alpha}{\sqrt{1 + (x + R/2)^2}} + \frac{Z_\beta}{\sqrt{1 + (x - R/2)^2}} \right]$$
    
    Attributes
    ----------
    eps : Float[Array, ""], 
        Float for numerical stability. 
    dim : Int[Scalar, ""]
        Dimension of the system, default is 1 dimension. 
    """
    
    eps: Float[Array, ""]
    dim: Int[Scalar, ""]

    def __init__(self, eps = 1e-5, dim = 1):
        self.eps = eps
        self.dim = dim 
    
    def __call__(
        self,
        x: Float[Array, "batch dim"],
        R: float,
        Z_alpha: int,
        Z_beta: int, 
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        r"""
        Compute 1D nuclear potential.
        
        Parameters
        ----------
        x : Float[Array, "batch dim"]
            Points where the potential is evaluated.
        R : float
            Distance between the two nuclei.
        Z_alpha : int
            Atomic number of the first nucleus.
        Z_beta : int
            Atomic number of the second nucleus.
        Ne : Int[Scalar, ""]
            Number of electrons.
            
        Returns
        -------
        Float[Array, "batch"]
            Attraction to the nuclei at each point x.
        """
        v_x = -Z_alpha / jnp.sqrt(1 + (x + R/2)**2) - Z_beta / jnp.sqrt(1 + (x - R/2)**2)
        return v_x * Ne