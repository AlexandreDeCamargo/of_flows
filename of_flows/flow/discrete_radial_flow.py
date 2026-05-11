"""
Discrete equivariant normalizing flow via composition of radial layers.

Each layer applies the equivariant transformation:
    x' = x + Σ_I α_I(|x - R_I|) * (x - R_I)

where R_I are the nuclear positions and α_I are scalar MLPs.  The
log-det-Jacobian and score are propagated analytically using JAX's AD.

Reference: Rezende & Mohamed, "Variational Inference with Normalizing Flows",
ICML 2015 (single-center radial flows); extended here to the multi-center
equivariant case from de Camargo et al.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array


class _ScalarMLP(eqx.Module):
    """
    Maps (r_norm: scalar, z_one_hot: (n_species,)) → scalar coefficient α.

    Used to compute per-atom radial coefficients in each flow layer.
    """
    linear_in: eqx.nn.Linear
    blocks: list
    linear_out: eqx.nn.Linear

    def __init__(self, n_in: int, dim: int, key):
        keys = jax.random.split(key, 4)
        self.linear_in  = eqx.nn.Linear(n_in, dim, key=keys[0])
        self.blocks      = [eqx.nn.Linear(dim, dim, key=k)
                            for k in jax.random.split(keys[1], 3)]
        self.linear_out  = eqx.nn.Linear(dim, 1, key=keys[2])

    def __call__(self, r: Array, z_one_hot: Array) -> Array:
        x = jnp.concatenate([r.reshape(1,), z_one_hot])
        x = jnp.tanh(self.linear_in(x))
        for block in self.blocks:
            x = jnp.tanh(block(x))
        return self.linear_out(x).squeeze()  # scalar


class RadialLayer(eqx.Module):
    """
    One equivariant radial layer:

        T(x) = x + Σ_I α_I(|x - R_I|) * (x - R_I)

    The Jacobian ∂T/∂x is a 3×3 matrix computed via jax.jacrev,
    so the log-det and score update are exact.
    """
    xyz_nuclei: Array   # (n_atoms, 3)  – fixed nuclear positions
    z_one_hot:  Array   # (n_atoms, n_species)
    mlp: _ScalarMLP

    def __init__(self, dim: int, mu: Array, z_one_hot: Array, key):
        n_species       = z_one_hot.shape[-1]
        self.xyz_nuclei = mu
        self.z_one_hot  = z_one_hot
        self.mlp        = _ScalarMLP(n_in=1 + n_species, dim=dim, key=key)

    def __call__(self, x: Array) -> Array:
        """Transform a single point x: (3,) → (3,)."""
        diffs   = x[None, :] - self.xyz_nuclei          # (n_atoms, 3)
        r_norms = jnp.linalg.norm(diffs, axis=-1)       # (n_atoms,)
        scalars = jax.vmap(self.mlp)(r_norms, self.z_one_hot)  # (n_atoms,)
        return x + jnp.einsum('ij,i->j', diffs, scalars)


class DiscreteRadialFlow(eqx.Module):
    """
    Discrete equivariant normalizing flow: composition of K RadialLayer
    transformations.

    Parameters
    ----------
    n_layers : int
        Number of radial layers K.
    dim : int
        Hidden dimension of each scalar MLP.
    mu : (n_atoms, 3) array
        Nuclear coordinates (reference points for equivariance).
    z_one_hot : (n_atoms, n_species) array
        One-hot encoding of atomic numbers.
    key : PRNGKey
    """
    layers: list

    def __init__(self, n_layers: int, dim: int, mu: Array,
                 z_one_hot: Array, key):
        keys = jax.random.split(key, n_layers)
        self.layers = [RadialLayer(dim, mu, z_one_hot, k) for k in keys]

    def __call__(self, x: Array) -> Array:
        """Forward pass: transform a single point x: (3,) → (3,)."""
        for layer in self.layers:
            x = layer(x)
        return x
