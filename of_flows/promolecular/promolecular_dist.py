from typing import Optional
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from distrax import MultivariateNormalDiag, Categorical
from jaxtyping import Array, Float, Int, Scalar
from dft_distrax import DFTDistribution
from pyscf import gto, dft
import jax.random as jrnd

AAtoBohr = 1.8897259886

class ProMolecularDensity(distrax.Distribution):
    r"""
        Creates a distribution for a molecule with a mixture of Gaussian components.

        Attributes
        ----------
        z : Int[Scalar, ""]
            Atomic numbers of the atoms in the molecule.
        dim : Int[Scalar, ""]
            Dimension of the system, default is 3 dimensions.
        loc : Float[Array, "z dim"]
            Molecular coordinates.
        scale_diag : Optional[Array], optional
            Sigma matrix, by default None
        units : str, optional
            Interatomic unit distance, by default 'Bohr'

    """        
    
    def __init__(
        self, 
        z: Int[Scalar, ""],
        loc: Float[Array, "z dim"],
        dim: Int[Scalar, ""] = 3,
        scale_diag: Optional[Array]=None,
        units: str = 'Bohr',
    ):       
        self.dim = dim 
        self.loc = lax.expand_dims(loc, dimensions=(1,))
        self.units = units

        if scale_diag is None:
            self.scale_diag = jnp.ones_like(self.loc)
        else:
            self.scale_diag = lax.expand_dims(scale_diag, dimensions=(1))

        if self.units.lower() == 'aa' or self.units.lower() == 'angstrom':
            self.loc = self.loc*AAtoBohr
            self.scale_diag = self.scale_diag*AAtoBohr

        self.logits = z
        self.probs = z/jnp.linalg.norm(z, ord=1)
        self.mixture_dist = Categorical(probs=self.probs)
        self.mixture_probs = self.mixture_dist.probs
        self.components_dist = MultivariateNormalDiag(
            loc=self.loc, scale_diag=self.scale_diag)

    @jax.jit
    def prob(self, value):
        log_px_components_dist = self.components_dist.log_prob(value).T
        px_components_dist = jnp.exp(log_px_components_dist)
        px = px_components_dist@self.mixture_probs[:, None]
        return px

    @jax.jit
    def log_prob(self, value):
        return jnp.log(self.prob(value))

    def _sample_n(self, key, n):
        _, key_mixt, key_comp = jax.random.split(key, 3)
        samples_mixt = self.mixture_dist._sample_n(key_mixt, n)
        samples_mixt_one_hot = jax.nn.one_hot(
            samples_mixt, self.mixture_probs.shape[-1])

        samples_comp = self.components_dist.sample(
            seed=key_comp, sample_shape=n)
        samples_comp = jnp.squeeze(samples_comp, axis=-2)

        samples = jnp.einsum('ijl,ij->il', samples_comp, samples_mixt_one_hot)
        return samples

    def event_shape(self):
        pass

    @jax.jit
    def score(self, values):
        return jax.vmap(jax.grad(lambda x:
                                 self.log_prob(x).sum()))(values)


class RadialDensityDistribution:
    """Distribution based on radial density sampling."""
    
    def __init__(self, db_prior, z, coords, grid_range=(-3.01, 3.0), n_grid_points=1000):
        self.db_prior = db_prior
        self.z = z
        self.coords = coords
        self.n_grid_points = n_grid_points

        x = jnp.linspace(-3, 3, 100)
        y = jnp.linspace(-3, 3, 100)

        X, Y = jnp.meshgrid(x, y)
        Z = jnp.zeros_like(X)

        points = jnp.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        # Pre-compute the radial grid
        self.rad_grid = jnp.linspace(grid_range[0], grid_range[1], n_grid_points)
        
        # Compute density along z-axis
        grid_points = jnp.array([
            jnp.zeros_like(self.rad_grid), 
            jnp.zeros_like(self.rad_grid), 
            self.rad_grid
        ]).T
        
        self.promol_dens = self.db_prior.density(grid_points)
        # self.p = self.promol_dens
        self.p = self.promol_dens /jnp.sum(self.promol_dens)
        
        # Pre-compute gradient
        self.promol_grad = self.db_prior.gradient(grid_points)
        
        # Compute score (gradient / density)
        self.promol_score = self.promol_grad / self.promol_dens.reshape(-1, 1)
        
        # Store last sampled indices for efficient lookup
        self._last_indices = None
    
    def sample(self, seed, sample_shape):
        """Sample positions from the radial density."""
        if isinstance(sample_shape, int):
            n_samples = sample_shape
        else:
            n_samples = sample_shape[0] if len(sample_shape) > 0 else 1
        
        # Sample indices according to density - shape (n_samples, 3)
        sampled_indices = jax.random.choice(
            seed, 
            a=len(self.rad_grid),
            shape=(n_samples, 3),
            replace=True,
            p=self.p
        )
        
        # Store indices for later lookup
        self._last_indices = sampled_indices
        
        # Map indices to radial values - shape (n_samples, 3)
        samples = self.rad_grid[sampled_indices]
        return samples
    
    def log_prob(self, value):
        """Compute log probability of samples."""
        # Find which grid index each value corresponds to
        # Since samples come from rad_grid, find closest match
        indices = jnp.argmin(jnp.abs(value[:, 2:3] - self.rad_grid[None, :]), axis=1)
        
        # Get densities for those indices
        densities = self.p[indices]
        log_probs = jnp.log(densities)
        
        return log_probs[:, None]  # Shape (batch_size, 1)
    
    def score(self, values):
        """Compute score (gradient of log probability)."""
        # Find which grid index each value corresponds to
        indices = jnp.argmin(jnp.abs(values[:, 2:3] - self.rad_grid[None, :]), axis=1)
        
        # Get pre-computed scores for those indices
        scores = self.promol_score[indices]
        
        return scores  # Shape (batch_size, 3)
    

# class DFTGridDistribution:
#     """Distribution based on PySCF grid with atomdb density."""
    
#     def __init__(self, db_prior, atoms, coords, basis='6-31G(d,p)', grid_level=3):
#         self.db_prior = db_prior
#         self.atoms = atoms
#         self.coords = coords
        
#         # Build atom string for PySCF (in Bohr units)
#         atom_string = ""
#         for atom, coord in zip(atoms, coords):
#             atom_string += f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}; "
        
#         # Create PySCF molecule and grid
#         mol = gto.M(atom=atom_string,
#                     basis=basis,
#                     unit='B',
#                     spin=0)
        
        
#         pyscfgrid = dft.gen_grid.Grids(mol)
#         pyscfgrid.level = grid_level
#         pyscfgrid.build()

        
#         self.grid_coords = jnp.array(pyscfgrid.coords)  
#         self.grid_weights = jnp.array(pyscfgrid.weights)  
        
#         self.promol_dens = self.db_prior.density(self.grid_coords)
#         self.promol_grad = self.db_prior.gradient(self.grid_coords)
#         self.promol_score = self.promol_grad / self.promol_dens.reshape(-1, 1)
        
        
#         # ADD THIS: Normalize density for sampling
#         # weighted_dens = self.promol_dens * self.grid_weights
#         # self.p = weighted_dens / jnp.sum(weighted_dens)
#         self.p = self.promol_dens /jnp.sum(self.promol_dens)
        
    
#     def sample(self, seed, sample_shape):
#         """Sample grid points weighted by density."""
#         if isinstance(sample_shape, int):
#             n_samples = sample_shape
#         else:
#             n_samples = sample_shape[0] if len(sample_shape) > 0 else 1
               
#         indices = jax.random.choice(
#             seed,
#             a=len(self.grid_coords),
#             shape=(n_samples,),
#             # replace=True,
#             p=self.p 
#         )
        
#         return self.grid_coords[indices] 
#     def log_prob(self, value):
#         """Compute log probability at points."""
#         distances = jnp.linalg.norm(
#             value[:, None, :] - self.grid_coords[None, :, :],
#             axis=2
#         )
#         indices = jnp.argmin(distances, axis=1)
        
#         probs =  self.promol_dens[indices]
#         log_probs = jnp.log(probs)
        
#         return log_probs[:, None]
    
#     def score(self, values):
#         """Compute score at points."""
#         distances = jnp.linalg.norm(
#             values[:, None, :] - self.grid_coords[None, :, :],
#             axis=2
#         )
#         indices = jnp.argmin(distances, axis=1)
        
#         return self.promol_score[indices]
    
class AtomDBDistribution:
    """Distribution based on atomdb density (no grid required)."""
    
    def __init__(self, db_prior, z, coords,Ne):
        self.db_prior = db_prior
        self.z = z
        self.coords = coords
        self.Ne = Ne
    
    def log_prob(self, value):
        """Compute log probability directly from atomdb density."""
        density = self.db_prior.density(value)
        normalized_density = density / self.Ne
        log_probs = jnp.log(normalized_density)
        return log_probs[:, None]
    
    def prob(self, value):
        """Compute log probability directly from atomdb density."""
        density = self.db_prior.density(value)
        normalized_density = density / self.Ne
        return normalized_density
    
    def score(self, values):
        """Compute score = gradient / density."""
        density = self.db_prior.density(values) / self.Ne
        gradient = self.db_prior.gradient(values)
        
        score = gradient / density.reshape(-1, 1)
        # score = jnp.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
  
        return score

class SIRDistribution:
    """
    Sampling Importance Resampling (SIR) distribution.
    Uses a base distribution as proposal and reweights to target distribution.
    """
    
    def __init__(
        self,
        base_distribution,      # ProMolecularDensity (Gaussian mixture)
        target_distribution,    # AtomDBDistribution (atomdb)
        oversampling_factor: int = 10
    ):
        """
        Parameters
        ----------
        base_distribution : ProMolecularDensity
            Base distribution (Gaussian mixture) - easy to sample from
        target_distribution : DFTGridDistribution
            Target distribution (atomdb) - what we actually want
        oversampling_factor : int
            How many proposal samples per final sample (higher = better quality)
        """
        self.base_distribution = base_distribution
        self.target_distribution = target_distribution
        self.oversampling_factor = oversampling_factor
    
    def sample(self, seed, sample_shape):
        """
        Sample using SIR: sample from base, reweight, resample.
        """
        if isinstance(sample_shape, int):
            n_final_samples = sample_shape
        else:
            n_final_samples = sample_shape[0] if len(sample_shape) > 0 else 1
        
        key_sample, key_resample = jrnd.split(seed)
        
        # Generate oversampled proposals from base distribution
        n_proposal_samples = self.oversampling_factor * n_final_samples
        
        proposal_samples = self.base_distribution.sample(
            seed=key_sample, 
            sample_shape=n_proposal_samples
        )
        
        # Compute importance weights
        density_base = self.base_distribution.prob(proposal_samples)  # Shape: (n_proposal_samples,) or (n_proposal_samples, 1)
        density_target = self.target_distribution.prob(proposal_samples)
        log_density_target = self.target_distribution.log_prob(proposal_samples)  # Shape: (n_proposal_samples, 1)
        
        # Flatten to 1D if needed
        density_base = density_base.squeeze()  # Shape: (n_proposal_samples,)
        
        log_density_target = log_density_target.squeeze()  # Shape: (n_proposal_samples,)
        density_target = jnp.exp(log_density_target)  # Shape: (n_proposal_samples,)
        
        # Importance weights: w = p_target / p_base
        importance_weights = density_target / (density_base)
        importance_weights = jnp.nan_to_num(importance_weights, nan=0.0, posinf=0.0, neginf=0.0)
        importance_weights = jnp.maximum(importance_weights, 0.0)
        importance_weights = importance_weights / jnp.sum(importance_weights)
        
        
        # Resample according to weights
        resampled_indices = jrnd.choice(
            key_resample,
            a=n_proposal_samples,
            shape=(n_final_samples,),
            p=importance_weights,
            replace=True
        )

        # Step 4: Calculate effective sample size
        effective_sample_size = 1.0 / jnp.sum(importance_weights**2)
        # Step 5: Resample according to importance weights
        if effective_sample_size < n_final_samples:
            print(f"Warning: Low effective sample size ({effective_sample_size:.1f}) for {n_final_samples} final samples")
            
        final_samples = proposal_samples[resampled_indices]
        return final_samples
    
    def log_prob(self, value):
        """
        Log probability uses the target distribution.
        """
        return self.target_distribution.log_prob(value)
    
    def score(self, values):
        """
        Score uses the target distribution.
        """
        return self.target_distribution.score(values)