"""
CNF analysis script: normalization, energy, binding energy, density plot.

Usage
-----
# Single molecule (energy + density):
  python test_fwd_rev.py \
      --results_dir of_flows/Results/H2/tf_w_lam0.2_none_lda_none_dopri8_promolecular_sched_MIX/bl_3.0000

# With binding energy (needs H atom result dir):
  python test_fwd_rev.py \
      --results_dir of_flows/Results/H2/tf_w_lam0.2_none_lda_none_dopri8_promolecular_sched_MIX/bl_3.0000 \
      --atom_results_dir of_flows/Results/H/tf_w_lam0.2_none_lda_none_dopri8_promolecular_sched_MIX/bl_0.0000
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "of_flows"))

import argparse
import glob
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrnd
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, dft
from atomdb import make_promolecule

jax.config.update("jax_enable_x64", True)

from flow.equiv_flows import CNF
from ode_solver.eqx_ode import fwd_ode, rev_ode
from utils import one_hot_encode, coordinates, get_solver
from promolecular.promolecular_dist import ProMolecularDensity, AtomDBDistribution, SIRDistribution
from train.loss import FUNCTIONAL_CLASSES, _build_kinetic
from functionals.functional import FunctionalInputs

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, required=True,
                    help="Path to bl_X.XXXX result directory (contains job_params.json)")
parser.add_argument("--atom_results_dir", type=str, default=None,
                    help="Path to the H atom bl_0.0000 result directory (for binding energy)")
parser.add_argument("--bs",         type=int, default=256,  help="Grid chunk size")
parser.add_argument("--grid_level", type=int, default=3,    help="PySCF grid level")
args = parser.parse_args()

# ── helpers ───────────────────────────────────────────────────────────────────
def load_results(results_dir: str):
    """Load job_params, find last checkpoint, build and restore the CNF model."""
    rdir = Path(results_dir).resolve()

    with open(rdir / "job_params.json") as f:
        p = json.load(f)

    # Find the highest-epoch checkpoint
    ckpts = glob.glob(str(rdir / "Checkpoints" / "checkpoint_*.eqx"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {rdir}/Checkpoints/")
    ckpts.sort(key=lambda path: int(re.search(r'checkpoint_(\d+)\.eqx', path).group(1)))
    last_ckpt = ckpts[-1]
    print(f"  Loading checkpoint: {last_ckpt}")

    Ne, atoms, z, coords = coordinates(p['mol_name'], p['bond_length'])
    z_one_hot = one_hot_encode(z)
    key = jrnd.PRNGKey(0)
    model = CNF(din=3, dim=p['hidden_layer'], mu=coords, one_hot=z_one_hot, key=key)
    model = eqx.tree_deserialise_leaves(last_ckpt, model)
    solver = get_solver(p['solver'])

    return p, model, solver, Ne, atoms, z, coords


def build_prior(p, z, coords, Ne):
    prior = ProMolecularDensity(z.ravel(), coords)
    if p['prior'] == 'db_sir':
        # Direct AtomDB sampling via per-atom inverse-CDF (no SIR needed)
        db_prior = make_promolecule(atnums=z, coords=coords, dataset="slater")
        return AtomDBDistribution(db_prior=db_prior, z=z, coords=coords, Ne=Ne)
    return prior


def build_pyscf_mol(atoms, coords, Ne):
    atom_str = "; ".join(f"{a} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}"
                         for a, c in zip(atoms, coords))
    return gto.M(atom=atom_str, basis="6-31G(d,p)", unit="B",
                 verbose=0, spin=int(Ne) % 2)


def compute_rho_on_grid(model, solver, sampling_dist, grid_coords, chunk):
    """Two-pass rev→fwd: get ρ and score at every grid point."""
    x_list, rho_list, score_list = [], [], []
    G = grid_coords.shape[0]
    for i in range(0, G, chunk):
        xc = grid_coords[i:i+chunk]
        n  = xc.shape[0]
        state_rev = jnp.concatenate([xc, jnp.zeros((n,1)), jnp.zeros((n,3))], axis=1)
        z_base, _ = rev_ode(model, state_rev, solver)
        log_p0    = sampling_dist.log_prob(z_base)
        score_p0  = sampling_dist.score(z_base)
        state_fwd = jnp.concatenate([z_base, log_p0, score_p0], axis=1)
        x_t1, logp_t1, score_t1 = fwd_ode(model, state_fwd, solver)
        x_list.append(np.array(x_t1))
        rho_list.append(np.array(jnp.exp(logp_t1)).ravel())
        score_list.append(np.array(score_t1))
    return (np.concatenate(x_list),
            np.concatenate(rho_list),
            np.concatenate(score_list))


def compute_energy(p, x_np, rho_np, score_np, grid_coords, grid_weights, mol_dict, Ne, chunk):
    """Quadrature integrals for all energy components."""
    rho_col = rho_np[:, None]
    w       = np.array(grid_weights)
    gc      = x_np

    t_func  = _build_kinetic(p['kinetic'], p['lam'])
    x_func  = FUNCTIONAL_CLASSES[p['exchange']]()
    n_func  = FUNCTIONAL_CLASSES[p['external']]()
    h_func  = FUNCTIONAL_CLASSES[p['hartree']]()
    c_func  = FUNCTIONAL_CLASSES[p['correlation']]() if p['correlation'] != 'none' else None
    cc_func = FUNCTIONAL_CLASSES[p['core_correction']]() if p['core_correction'] != 'none' else None

    G = rho_np.shape[0]
    t_e = np.zeros(G); x_e = np.zeros(G)
    n_e = np.zeros(G); c_e = np.zeros(G); cc_e = np.zeros(G)

    for i in range(0, G, chunk):
        sl  = slice(i, min(i+chunk, G))
        inp = FunctionalInputs(den=jnp.array(rho_col[sl]), score=jnp.array(score_np[sl]),
                               x=jnp.array(gc[sl]), Ne=Ne, mol=mol_dict, xp=None)
        t_e[sl]  = np.array(t_func(inp)).ravel()
        x_e[sl]  = np.array(x_func(inp)).ravel()
        n_e[sl]  = np.array(n_func(inp)).ravel()
        if c_func  is not None: c_e[sl]  = np.array(c_func(inp)).ravel()
        if cc_func is not None: cc_e[sl] = np.array(cc_func(inp)).ravel()

    T   = float(np.dot(w, t_e  * rho_np))
    E_X = float(np.dot(w, x_e  * rho_np))
    V_N = float(np.dot(w, n_e  * rho_np))
    E_C = float(np.dot(w, c_e  * rho_np))
    E_CC= float(np.dot(w, cc_e * rho_np))

    # Hartree — O(G²) double integral, j≠k
    coords_H  = np.array(grid_coords)
    v_coulomb = np.zeros(G)
    for i in range(0, G, chunk):
        xi       = coords_H[i:i+chunk]
        diff     = coords_H[None,:,:] - xi[:,None,:]
        r2       = np.sum(diff**2, axis=-1)
        safe_r   = np.sqrt(np.where(r2 == 0., np.inf, r2))
        v_coulomb[i:i+chunk] = np.dot(1./safe_r, w * rho_np)
    V_H = float(0.5 * Ne**2 * np.dot(w * rho_np, v_coulomb))

    # Nuclear repulsion
    coords_np = np.array(mol_dict['coords'])
    z_arr     = np.array(mol_dict['z']).ravel()
    E_NN = 0.0
    for I in range(len(coords_np)):
        for J in range(I+1, len(coords_np)):
            E_NN += float(z_arr[I]) * float(z_arr[J]) / float(np.linalg.norm(coords_np[I]-coords_np[J]))

    E_total = T + V_N + V_H + E_X + E_C + E_CC + E_NN
    return dict(T=T, V_N=V_N, V_H=V_H, E_X=E_X, E_C=E_C, E_CC=E_CC, E_NN=E_NN,
                E_total=E_total)


def run_analysis(results_dir: str, chunk: int, grid_level: int):
    """Full analysis for one result directory. Returns energy dict."""
    print(f"\n{'='*60}")
    print(f"Analysing: {results_dir}")
    p, model, solver, Ne, atoms, z, coords = load_results(results_dir)
    mol_dict   = {'coords': coords, 'z': z}
    sampling_dist = build_prior(p, z, coords, Ne)

    mol_pyscf = build_pyscf_mol(atoms, coords, Ne)
    grid = dft.gen_grid.Grids(mol_pyscf)
    grid.level = grid_level
    grid.build()
    grid_coords  = jnp.array(grid.coords,  dtype=jnp.float64)
    grid_weights = jnp.array(grid.weights, dtype=jnp.float64)
    print(f"  Grid: {grid_coords.shape[0]} points  (level={grid_level})")

    print("  Computing ρ via rev→fwd ...")
    x_np, rho_np, score_np = compute_rho_on_grid(
        model, solver, sampling_dist, grid_coords, chunk)

    pos_err = float(np.max(np.abs(x_np - np.array(grid_coords))))
    Ne_est  = float(np.dot(np.array(grid_weights), Ne * rho_np))
    print(f"  Round-trip error : {pos_err:.3e}")
    print(f"  ∫ρ_M dx          : {Ne_est:.6f}  (should be {Ne})")

    energies = compute_energy(
        p, x_np, rho_np, score_np,
        grid_coords, grid_weights, mol_dict, Ne, chunk)

    print(f"\n  === ENERGY  ({p['kinetic']} / λ={p['lam']} / {p['exchange']}) ===")
    print(f"  T     = {energies['T']:+.6f} Ha")
    print(f"  V_N   = {energies['V_N']:+.6f} Ha")
    print(f"  V_H   = {energies['V_H']:+.6f} Ha")
    print(f"  E_X   = {energies['E_X']:+.6f} Ha")
    if p['correlation']    != 'none': print(f"  E_C   = {energies['E_C']:+.6f} Ha")
    if p['core_correction']!= 'none': print(f"  E_CC  = {energies['E_CC']:+.6f} Ha")
    if energies['E_NN'] != 0.0:       print(f"  E_NN  = {energies['E_NN']:+.6f} Ha")
    print(f"  ─────────────────────")
    print(f"  E_tot = {energies['E_total']:+.6f} Ha")
    if p['mol_name'] == 'H':
        print(f"  (exact H = -0.500000 Ha)")

    return p, model, solver, Ne, atoms, coords, grid_coords, grid_weights, \
           rho_np, score_np, sampling_dist, energies


# ── Main molecule ─────────────────────────────────────────────────────────────
(p, model, solver, Ne, atoms, coords, grid_coords, grid_weights,
 rho_np, score_np, sampling_dist, energies) = run_analysis(
    args.results_dir, args.bs, args.grid_level)

# ── Atom reference for binding energy ─────────────────────────────────────────
if args.atom_results_dir is not None:
    _, _, _, _, _, _, _, _, _, _, _, energies_H = run_analysis(
        args.atom_results_dir, args.bs, args.grid_level)

    E_mol  = energies['E_total']
    E_atom = energies_H['E_total']
    E_bind = E_mol - 2.0 * E_atom          # negative = bound
    D_e    = -E_bind                        # dissociation energy (positive = stable)

    print(f"\n=== BINDING ENERGY ===")
    print(f"  E({p['mol_name']}, R={p['bond_length']:.4f} Bohr) = {E_mol:+.6f} Ha")
    print(f"  E(H atom)                           = {E_atom:+.6f} Ha")
    print(f"  E_bind = E(mol) - 2·E(H)            = {E_bind:+.6f} Ha")
    print(f"  D_e    = 2·E(H) - E(mol)            = {D_e:+.6f} Ha  ({D_e*27.2114:.4f} eV)")

# ── Density plot along z-axis ─────────────────────────────────────────────────
z_min = float(coords[:, 2].min()) - 3.0
z_max = float(coords[:, 2].max()) + 3.0
zt    = np.linspace(z_min, z_max, 300)
line_pts = jnp.array(np.stack([np.zeros_like(zt),
                                np.zeros_like(zt),
                                zt], axis=1), dtype=jnp.float64)

print("\nComputing density along z-axis ...")
rho_line = []
for i in range(0, line_pts.shape[0], args.bs):
    xc = line_pts[i:i+args.bs]
    n  = xc.shape[0]
    state_rev = jnp.concatenate([xc, jnp.zeros((n,1)), jnp.zeros((n,3))], axis=1)
    z_b, _    = rev_ode(model, state_rev, solver)
    log_p0    = sampling_dist.log_prob(z_b)
    score_p0  = sampling_dist.score(z_b)
    _, logp_fwd, _ = fwd_ode(model,
        jnp.concatenate([z_b, log_p0, score_p0], axis=1), solver)
    rho_line.append(np.array(jnp.exp(logp_fwd)).ravel())
rho_pred = np.concatenate(rho_line)

R = float(jnp.linalg.norm(coords[0] - coords[-1])) if len(coords) > 1 else 0.0

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(zt, Ne * rho_pred, color='tab:blue',
        label=rf"$N_e\,\rho_{{NF}}(z)$,  R={R:.3f} Bohr")
ax.set_xlabel("z [Bohr]")
ax.set_ylabel(r"$\rho(z)$ [Bohr$^{-3}$]")
ax.set_title(f"{p['mol_name']}  |  {p['kinetic']} λ={p['lam']}  |  {p['exchange']}")
ax.legend()
fig.tight_layout()

out_dir = Path(args.results_dir).resolve()
fig.savefig(out_dir / "density.svg", transparent=True)
fig.savefig(out_dir / "density.png", dpi=150)
print(f"Density plot saved → {out_dir}/density.png")

# ── Save energy summary ───────────────────────────────────────────────────────
summary = {**energies,
           'mol_name':    p['mol_name'],
           'bond_length': p['bond_length'],
           'Ne_integral': float(np.dot(np.array(grid_weights), Ne * rho_np))}
if args.atom_results_dir is not None:
    summary['E_atom'] = energies_H['E_total']
    summary['E_bind'] = E_bind
    summary['D_e']    = D_e

with open(out_dir / "energy_summary.json", "w") as f:
    json.dump(summary, f, indent=4)
print(f"Energy summary saved → {out_dir}/energy_summary.json")
