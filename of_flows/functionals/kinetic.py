import jax 
from jax import lax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Scalar

class ThomasFermi(eqx.Module):
    r"""
    Thomas-Fermi kinetic functional.
    
    $$T_{\text{TF}}[\rho] = c \int \rho^{5/3} d\mathbf{x}$$
    
    See paper eq. 2 in https://pubs.aip.org/aip/jcp/article/114/2/631/184186
    
    Attributes
    ----------
    c : Float[Array, ""]
        Prefactor constant, default is (3/10)(3\pi^2)^(2/3).
    dim : Int[Scalar, ""]
        Dimension of the system, default is 3 dimensions.
    """
    
    c: Float[Array, ""]
    dim: Int[Scalar, ""]
    
    def __init__(
        self,
        c = (3./10.) * (3.*jnp.pi**2)**(2/3), 
        dim = 3
    ):
        self.c = c
        self.dim = dim 
    
    def __call__(
        self,
        den: Float[Array, "batch"],
        score: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        r"""
        Compute Thomas-Fermi kinetic energy density.
        
        Parameters
        ----------
        den : Float[Array, "batch"]
            Electron density at each spatial point.
        score : Float[Array, "batch dim"]
            Gradient of log-likelihood (score function).
        Ne : Int[Scalar, ""]
            Number of electrons.
            
        Returns
        -------
        Float[Array, "batch"]
            Kinetic energy density at each batch point.
        """
        val = den**(2/3)
        return self.c * (Ne**(5/3)) * val


class ThomasFermi1D(eqx.Module):
    r"""
    Thomas-Fermi kinetic functional in 1D.
    
    $$T_{\text{TF}}[\rho] = \frac{\pi^2}{24} \int \rho^3 dx$$
    
    See original paper eq. 18 in https://pubs.aip.org/aip/jcp/article/139/22/224104/193579
    
    Attributes
    ----------
    c : Float[Array, ""]
        Prefactor constant, default is \pi^2/24.
    dim : Int[Scalar, ""]
        Dimension of the system, default is 1 dimension. 
    """
    
    c: Float[Array, ""]
    dim: Int[Scalar, ""]
    
    def __init__(
        self, 
        c = (jnp.pi**2) / 24, 
        d = 1
    ):
        self.c = c
        self.d = d 
    
    def __call__(
        self,
        den: Float[Array, "batch"],
        score: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        r"""
        Compute Thomas-Fermi 1D kinetic energy density.
        
        Parameters
        ----------
        den : Float[Array, "batch d"]
            Electron density at each spatial point.
        score : Float[Array, "batch d"]
            Gradient of log-likelihood (score function).
        Ne : Int[Scalar, ""]
            Number of electrons.
            
        Returns
        -------
        Float[Array, "batch d"]
            Kinetic energy density at each batch point.
        """
        den_sqr = den * den
        return self.c * (Ne**3) * den_sqr
    

class Weizsacker(eqx.Module):
    r"""
    von Weizsäcker gradient correction.
    
    $$T_W[\rho] = \frac{\lambda}{8} \int \frac{(\nabla \rho)^2}{\rho} d\mathbf{x}$$
    
    See paper eq. 3 in https://pubs.aip.org/aip/jcp/article/114/2/631/184186
    
    Attributes
    ----------
    lambda_0 : float, optional (W Stich, EKU Gross., Physik A Atoms and Nuclei, 309(1):511, 1982.)
        Phenomenological parameter, default is 0.2.
    d : Int[Scalar, ""]
        Dimension of the system, default is 3 dimensions. 
    """
    
    lambda_0: Float[Array, ""]
    dim: Int[Scalar, ""]


    def __init__(
        self, 
        lambda_0 = 0.2, 
        dim = 3
    ):
        self.lambda_0 = lambda_0
        self.dim = dim 
    
    def __call__(
        self,
        den: Float[Array, "batch"],
        score: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        r"""
        Compute Weizsäcker kinetic energy density.
        
        Parameters
        ----------
        den : Float[Array, "batch"]
            Electron density at each spatial point.
        score : Float[Array, "batch dim"]
            Gradient of log-likelihood (score function).
        Ne : Int[Scalar, ""]
            Number of electrons.
            
        Returns
        -------
        Float[Array, "batch"]
            Kinetic energy density at each batch point.
        """
        score_sqr = jnp.einsum('ij,ij->i', score, score)
        return (self.lambda_0*Ne/8.)*lax.expand_dims(score_sqr, (1,))

        


