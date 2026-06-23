"""
Potential Energy Surface scan over a set of bond-length result directories.

Thin CLI around ``of_flows/quadrature.py``: grid-integrates every bl_* directory
under --scan_dir (via ``grid_energy_from_checkpoint``) and, optionally, an atom
reference for the binding energy, then writes pes.csv and a plot.

Usage
-----
  python scan_pes.py \
      --scan_dir  Results/H2/<method> \
      --atom_dir  Results/H/<method>/bl_0.0000

Outputs (written inside --scan_dir):
  pes.csv          — R, E_total, T, V_N, V_H, E_X, E_NN, E_bind, D_e  (Ha / eV)
  pes.png / .svg   — PES curve (E_total and D_e vs R)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from quadrature import grid_energy_from_checkpoint

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--scan_dir",  type=str, required=True,
                    help="Method directory containing bl_X.XXXX subdirectories")
parser.add_argument("--atom_dir",  type=str, default=None,
                    help="bl_0.0000 directory for the atom (binding-energy reference)")
parser.add_argument("--bs",        type=int, default=256,  help="Grid chunk size")
parser.add_argument("--grid_level",type=int, default=3,    help="PySCF grid level")
parser.add_argument("--recompute", action="store_true",
                    help="Re-run integration even if energy_summary.json already exists")
args = parser.parse_args()


def analyse(results_dir):
    return grid_energy_from_checkpoint(
        Path(results_dir).resolve(), grid_level=args.grid_level,
        chunk=args.bs, recompute=args.recompute)


# ── Scan over all bl_* directories ───────────────────────────────────────────
scan_dir = Path(args.scan_dir).resolve()
bl_dirs  = sorted(scan_dir.glob("bl_*"),
                  key=lambda d: float(d.name.split("_")[1]))
if not bl_dirs:
    raise FileNotFoundError(f"No bl_* directories found in {scan_dir}")

print(f"\nFound {len(bl_dirs)} bond-length directories in:\n  {scan_dir}\n")

# Optional atom reference (binding uses 2*E_atom — homonuclear diatomic)
E_atom = None
if args.atom_dir is not None:
    print("atom reference:")
    E_atom = analyse(args.atom_dir)['E_total']
    print(f"  E(atom) = {E_atom:+.6f} Ha\n")

# Main scan
rows = []
for bl_dir in bl_dirs:
    if not (bl_dir / "job_params.json").exists():
        print(f"  {bl_dir.name}: missing job_params.json — skipping")
        continue
    data = analyse(bl_dir)
    R = data['bond_length']
    row = {'R_bohr':  R,
           'epoch':   data.get('epoch', '?'),
           'E_total': data['E_total'],
           'T':       data['T'],
           'V_N':     data['V_N'],
           'V_H':     data['V_H'],
           'E_X':     data['E_X'],
           'E_C':     data.get('E_C', 0.0),
           'E_NN':    data['E_NN'],
           'Ne_int':  data['Ne_integral']}
    if E_atom is not None:
        E_bind           = data['E_total'] - 2.0 * E_atom
        row['E_bind_Ha'] = E_bind
        row['D_e_eV']    = -E_bind * 27.2114
    rows.append(row)
    tag = f"  R={R:.4f} Bohr  epoch={row['epoch']:>6}  E={data['E_total']:+.6f} Ha"
    if E_atom is not None:
        tag += f"  E_bind={row['E_bind_Ha']:+.6f} Ha"
    print(tag)

# ── Save CSV ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows).sort_values('R_bohr').reset_index(drop=True)
csv_path = scan_dir / "pes.csv"
df.to_csv(csv_path, index=False, float_format='%.8f')
print(f"\nPES data saved → {csv_path}")
print(df.to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2 if E_atom is not None else 1,
                         figsize=(11 if E_atom is not None else 5, 4))
if E_atom is None:
    axes = [axes]

R_vals    = df['R_bohr'].values
max_epoch = df['epoch'].max()
complete  = df['epoch'] == max_epoch   # True if run finished

for ax, y_col, ylabel, title, color in [
    (axes[0], 'E_total', 'Energy [Ha]',  'Potential Energy Surface', 'tab:blue'),
    *( [(axes[1], 'D_e_eV', 'D_e [eV]', 'Dissociation Energy', 'tab:orange')]
       if E_atom is not None else [] ),
]:
    y = df[y_col].values
    ax.plot(R_vals[complete],  y[complete],  'o-', color=color, label=f'epoch={max_epoch}')
    ax.plot(R_vals[~complete], y[~complete], '^', color=color, alpha=0.5,
            label='incomplete', markerfacecolor='none')
    if y_col == 'D_e_eV':
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel("R [Bohr]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(scan_dir.parent.name.split("/")[-1], fontsize=9)
fig.tight_layout()
fig.savefig(scan_dir / "pes.svg", transparent=True)
fig.savefig(scan_dir / "pes.png", dpi=150)
print(f"PES plot saved → {scan_dir}/pes.png")
