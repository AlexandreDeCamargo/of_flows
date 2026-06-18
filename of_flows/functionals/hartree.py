import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Scalar

class CoulombPotential(eqx.Module):
    r"""
    Classical electron-electron repulsion (Hartree) potential.
    
    $$V_{\text{Hartree}}(\mathbf{x}, \mathbf{x}') = \frac{1}{2} N_e^2 \frac{1}{|\mathbf{x} - \mathbf{x}'|}$$
    
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
    
    def __call__(
        self,
        x: Float[Array, "batch dim"],
        xp: Float[Array, "batch dim"], 
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        r"""
        Compute Coulomb potential.
        
        Parameters
        ----------
        x : Float[Array, "batch d"]
            Points where the potential is evaluated.
        xp : Float[Array, "batch d"] 
            Points where the charge density is located.
        Ne : Int[Scalar, ""]
            Number of electrons.
            
        Returns
        -------
        Float[Array, "batch"]
            Coulomb potential at each point x.
        """
        diff = x - xp
        z = jnp.sum(diff * diff + self.eps, axis=-1, keepdims=True)
        z = 1.0 / jnp.sqrt(z)
        return 0.5 * (Ne ** 2) * z.squeeze()


class CoulombPotentialAllPairs(eqx.Module):
    r"""
    Hartree potential over **all** sample pairs (lower-variance estimator).

    Same physics as CoulombPotential — true $1/|\mathbf{x}-\mathbf{x}'|$ Coulomb —
    but instead of pairing the two half-batches element-wise (x_i with x'_i, only
    `batch` pairs) it averages each x_i against *every* x'_j, i.e. the full
    batch×batch double sum. This is the same estimator the grid quadrature uses
    and has substantially lower Monte-Carlo variance for the same samples — it
    averages over batch² pairs instead of batch — at the cost of a (batch, batch)
    distance matrix.

    Returns the per-sample potential averaged over all x'_j, shape (batch,), so
    it is a drop-in replacement for CoulombPotential in the loss.
    """

    eps: Float[Array, ""]
    dim: Int[Scalar, ""]

    def __init__(self, eps=1e-5, dim=3):
        self.eps = eps
        self.dim = dim

    def __call__(
        self,
        x: Float[Array, "batch dim"],
        xp: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        # |x_i - xp_j|^2 via the expansion, avoiding the (batch, batch, dim) tensor
        x2  = jnp.sum(x * x,   axis=-1)                       # (batch,)
        xp2 = jnp.sum(xp * xp, axis=-1)                       # (batch,)
        r2  = x2[:, None] + xp2[None, :] - 2.0 * (x @ xp.T)   # (batch, batch)
        r2  = jnp.maximum(r2, 0.0) + self.eps                 # guard roundoff, regularise
        inv_r = 1.0 / jnp.sqrt(r2)                            # (batch, batch)
        v = jnp.mean(inv_r, axis=-1)                          # (batch,)  avg over all xp_j
        return 0.5 * (Ne ** 2) * v


class SoftCoulombPotential(eqx.Module):
    r"""
    We consider the a one-dimensional model, where the electron repulsion has the soft-Coulombic format. For more
    information see eq 6 in https://pubs.aip.org/aip/jcp/article/139/22/224104/193579/Orbital-free-bond-breaking-via-machine-learning
    
    Soft-Coulomb: 
    $$V_{\text{soft}}(\mathbf{x}, \mathbf{x}') = \frac{N_e^2}{\sqrt{1 + |\mathbf{x} - \mathbf{x}'|^2}}$$
    
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
        xp: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        r"""
        Compute soft Coulomb potential.
        
        Parameters
        ----------
        x : Float[Array, "batch d"]
            Points where the potential is evaluated.
        xp : Float[Array, "batch d"]
            Points where the charge density is located.
        Ne : Int[Scalar, ""]
            Number of electrons.
            
        Returns
        -------
        Float[Array, "batch"] 
            Soft Coulomb potential at each point x.
        """
        diff = x - xp
        dist_sq = jnp.sum(diff * diff, axis=-1)
        v_coul = 1.0 / jnp.sqrt(1.0 + dist_sq)
        return v_coul * (Ne ** 2)