import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Scalar
from jax import vmap, lax

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
                jnp.sum((x - molecule['coords']) ** 2, axis=1) + self.eps ** 2
            )
            z = molecule['z']
            w_i = jnp.exp(-self.a * z * r)
            return w_i

        _wi_mapped = vmap(_wi, in_axes=(None, 0), out_axes=-1)(x, molecule)
        
        score_sqr = jnp.einsum('ij,ij->i', score, score)
        weizs = (1.0 / 8.0) * lax.expand_dims(score_sqr, (1,))
        # weizs = (0.2 / 8.0) * lax.expand_dims(score_sqr, (1,))
        
        kinetic = jnp.abs(weizs - (molecule['z']**2 / 2))
        term = jnp.sum(kinetic * _wi_mapped, axis=-1, keepdims=True)
        
        return Ne * term


class HutcheonCuspCondition(eqx.Module):
    r"""
    Exact nuclear cusp cost functional from Hutcheon & Wibowo-Teale,
    Phys. Rev. B 110, 195146 (2024), Eqs. (4-5, 7).

    E_CC = Σ_i C_i,   C_i(ρ) = ∫ |∇ρ + 2Z_i ρ r̂_i|² w_i(r) d³r

    where  r̂_i = (r - R_i)/|r - R_i|
    and    w_i(r) = (π/Z_i³)^(1/2) exp(-Z_i |r - R_i|)   [Eq. (5)]

    Since ∇ρ = ρ · score, the integrand becomes ρ² |score + 2Z_i r̂_i|²,
    and the Kato cusp is satisfied exactly when score = -2Z_i r̂_i at R_i.

    Monte-Carlo estimate with samples from p = ρ/Ne:
      C_i ≈ Ne² · mean_k[ den(x_k) · |score(x_k) + 2Z_i r̂_i(x_k)|² · w_i(x_k) ]
    """

    eps: Float[Array, ""]

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "batch 3"],
        den: Float[Array, "batch 1"],
        score: Float[Array, "batch 3"],
        Ne: Int[Scalar, ""],
        mol: dict,
    ) -> Float[Array, "batch 1"]:
        coords = mol['coords']  # (n_atoms, 3)
        z = mol['z']            # (n_atoms,)

        def per_atom(R_i, Z_i):
            r_vec  = x - R_i                                               # (batch, 3)
            r_norm = jnp.sqrt(jnp.sum(r_vec ** 2, axis=-1) + self.eps**2) # (batch,)
            r_hat  = r_vec / r_norm[:, None]                               # (batch, 3)

            # indicator: 1s-orbital envelope, Eq. (5)
            w_i = jnp.sqrt(jnp.pi / Z_i ** 3) * jnp.exp(-Z_i * r_norm)   # (batch,)

            # exact Kato vector: zero iff cusp condition is satisfied
            cusp_vec = score + 2.0 * Z_i * r_hat                          # (batch, 3)
            cusp_sq  = jnp.sum(cusp_vec ** 2, axis=-1)                    # (batch,)

            return cusp_sq * w_i                                           # (batch,)

        # sum contributions of all nuclei  →  (batch,)
        atom_contribs = jax.vmap(per_atom)(coords, z)   # (n_atoms, batch)
        total = jnp.sum(atom_contribs, axis=0)           # (batch,)

        # ρ = Ne · den;  C_i per-sample = ρ · |...|² · w_i
        rho = Ne * den.squeeze(-1)                        # (batch,)
        cc  = rho * total                                 # (batch,)

        return cc[:, None]                                # (batch, 1)

