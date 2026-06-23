# OFF: Orbital-Free DFT with Normalizing Flows

OFF is a [JAX](https://github.com/google/jax)-based library for **orbital-free density
functional theory (OF-DFT)** in which the electron density is represented by a
**continuous normalizing flow (CNF)** and the ground-state energy is obtained by
*variationally minimizing* a density functional with Monte-Carlo gradient estimates.
The density is normalized by construction (it is a probability flow), and the physical
density is recovered as `ρ(x) = Ne · ρ_φ(x)`.

OFF is built entirely on the JAX ecosystem — automatic differentiation, JIT
compilation, vectorization, and GPU acceleration — with
[Diffrax](https://github.com/patrick-kidger/diffrax) for the flow ODEs,
[Equinox](https://github.com/patrick-kidger/equinox) for the models and functionals,
[Distrax](https://github.com/google-deepmind/distrax) for the base distribution, and
[Optax](https://github.com/google-deepmind/optax) for the optimization.

## Functionality

* Represent the electron density with a continuous normalizing flow; sample it, and
  evaluate both `ρ_φ(x)` and `∇log ρ_φ(x)` at arbitrary points.
* Promolecular base distribution (a Gaussian mixture), or an AtomDB-based prior
  (`--prior db_sir`, optional dependency).
* A modular library of density functionals — selected or extended without touching
  the flow:
  * **kinetic**: Thomas–Fermi, von Weizsäcker, and TF-λW;
  * **exchange**: Dirac/LDA and B88;
  * **correlation**: VWN and PW92;
  * **nuclear attraction**, **Hartree** (a low-variance all-pairs Monte-Carlo
    estimator, and an exact grid double-sum), and **nuclear-cusp corrections**
    (Kato, Hutcheon).
* GradDFT-style functional construction, `F[ρ] = ∫ c_θ[ρ](x)·e[ρ](x) dx`, assembled
  from `coefficients` and `energy_densities`; a single `EnergyFunctional` receives one
  shared input bundle and routes it to every component.
* Optimization with Optax: AdamW, gradient clipping, learning-rate schedules, and an
  exponential moving average of the noisy Monte-Carlo energy estimate.
* A deterministic **grid (quadrature) readout** of the energy after training, using a
  [PySCF](https://github.com/pyscf/pyscf) Becke grid converted to `jax.Array`.

## Install

A core dependency is [PySCF](https://pyscf.org), which needs `cmake` available on the
`PATH`. In a fresh environment, from the repository root:

```bash
pip install -e .
```

The `db_sir` prior additionally requires AtomDB; install it only if you use that prior.

## Use

Two stages: (1) **train** a normalizing flow for a given molecule and functional, then
(2) read out the energy on a **quadrature grid**. Both are exposed as command-line
tools (after `pip install -e .`) and as a Python API.

### 1. Train a flow

```bash
off-train --mol_name H2 --bond_length 1.4 \
          --kin tf_w --lam 1/5 --x lda_b88_x --c none \
          --hart coulomb --prior promolecular \
          --solver dopri8 --epochs 500 --bs 512
```
(equivalently `python -m off.main ...`). This minimizes the OF-DFT energy with the
Monte-Carlo estimator and writes everything under a method-tagged directory:

```
Results/H2/tf_w_lam0.2_none_lda_b88_x_none_dopri8_promolecular_sched_mix/bl_1.40/
    Checkpoints/checkpoint_*.eqx     # the trained flow
    training_metrics_ema.csv         # EMA energy trace
    job_params.json                  # everything needed to rebuild the run
```

Key options: `--kin {tf,w,tf_w}`, `--x {lda,b88_x,lda_b88_x}`, `--c {vwn_c,pw92_c,none}`,
`--cc {kato,hutcheon,none}`, `--hart {coulomb,coulomb_}` (all-pairs / element-wise),
`--prior {promolecular,db_sir}`, `--solver {dopri5,tsit5,dopri8}`.

### 2. Evaluate the energy on a grid

After training, point the quadrature tool at the run directory. It rebuilds the flow
from `job_params.json`, builds a PySCF grid, evaluates `ρ_φ` and its score there, and
integrates every energy term:

```bash
off-quadrature Results/H2/tf_w_lam0.2_none_lda_b88_x_none_dopri8_promolecular_sched_mix/bl_1.40 --grid_level 3
```
(equivalently `python -m off.quadrature ...`). It prints the per-term energies
(`T, V_N, V_H, E_X, E_C, E_CC, E_NN, E_total`) and the `∫ρ` check, and caches the
result in `energy_summary.json`. The same call from Python:

```python
from off import grid_energy_from_checkpoint

e = grid_energy_from_checkpoint(
    "Results/H2/.../bl_1.40", grid_level=3)
print(e["E_total"], e["Ne_integral"])
```

### Build a grid

```python
from off import getGrid

h2_geom = "H 0 0 0; H 0 0 1.4"               # PySCF atom= string
w_grid, x_grid = getGrid(h2_geom, level=3,
                         basis="6-31G(d,p)", units="bohr")   # -> (weights, coords)
```

### Define a functional

Every functional is `F[ρ] = ∫ c_θ[ρ]·e[ρ]`, built from a `coefficients` callable and an
`energy_densities` callable. Fixed functionals use the constant `unit_coefficient`
(`c = 1`); a learnable one would supply a network instead. Energy densities are returned
*up to the multiplicative ρ factor* (the Monte-Carlo expectation supplies it).

```python
import jax.numpy as jnp
from off import (Functional, unit_coefficient, EnergyFunctional,
                 tf_weizsacker, NuclearPotential, CoulombPotential)

# LDA exchange energy density (inp bundles den, score, x, Ne, mol, xp)
def lda_density(inp):
    l = -(3 / 4) * (3 / jnp.pi) ** (1 / 3) * (inp.Ne ** (4 / 3))
    return l * inp.den ** (1 / 3)

lda = Functional(coefficients=unit_coefficient, energy_densities=lda_density)

# Assemble the full energy functional; every component receives the same input bundle.
functional = EnergyFunctional(
    kinetic  = tf_weizsacker(lam=0.2),       # T_TF + 0.2 T_W
    external = NuclearPotential(),
    hartree  = CoulombPotential(),
    exchange = lda,
)
```

Prebuilt functionals are importable directly: `tf`, `weizsacker`, `tf_weizsacker`,
`lda`, `b88`, `vwn`, `pw92`, `lda_b88`, `NuclearPotential`, `CoulombPotential`,
`KatoCondition`, `HutcheonCuspCondition`.

## Package layout

```
off/
  main.py            # training entry point   (off-train)
  quadrature.py      # grid-energy readout    (off-quadrature)
  flow/              # the continuous normalizing flow (CNF)
  ode_solver/        # Diffrax forward/reverse ODE solves
  promolecular/      # base distributions (promolecular, AtomDB)
  functionals/       # kinetic, exchange-correlation, nuclear, Hartree, cusp
  train/             # loss (Monte-Carlo energy) and the optimization loop
```

## Citation

```bibtex
@article{off,
  title  = {Orbital-Free DFT with Normalizing Flows},
  author = {de Camargo, Alexandre and others},
  year   = {2026},
}
```
