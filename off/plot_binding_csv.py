"""
Plot a binding_{mol}.csv:  PES (left) and ΔE binding (right).
  MC   = blue line + markers
  grid = orange dots

Usage:
    python plot_binding_csv.py Results/N2/binding_N2.csv
    python plot_binding_csv.py Results/N2/binding_N2.csv --out n2.png
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv", help="path to binding_{mol}.csv")
ap.add_argument("--out", default=None, help="output image (default: <csv>.png)")
args = ap.parse_args()

df = pd.read_csv(args.csv)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

for method, g in df.groupby("method"):
    g = g.sort_values("R_bohr")

    # ── left: total energy (PES) ──────────────────────────────────────────────
    axL.plot(g.R_bohr, g.E_AB_mc, "-o", color="tab:blue", lw=2, ms=7, label="MC")
    axL.scatter(g.R_bohr, g.E_AB_grid, color="orange", marker="o", s=70,
                edgecolors="black", linewidths=0.5, zorder=5, label="grid")
    if "E_atoms_grid" in g.columns:                       # dissociation limit (grid)
        axL.axhline(g.E_atoms_grid.iloc[0], color="orange", ls=":", lw=1.2,
                    alpha=0.9, label=r"2·E(atom) grid")

    # ── right: ΔE = E(A) + E(B) - E(AB) ───────────────────────────────────────
    axR.plot(g.R_bohr, g.dE_mc_Ha, "-o", color="tab:blue", lw=2, ms=7, label="MC")
    axR.scatter(g.R_bohr, g.dE_grid_Ha, color="orange", marker="o", s=70,
                edgecolors="black", linewidths=0.5, zorder=5, label="grid")

axL.set_xlabel("R [Bohr]")
axL.set_ylabel(r"E[$\rho$] + V$_{NN}$(R) [a.u.]")
axL.set_title("PES")
axL.grid(alpha=0.3)
axL.legend(fontsize=8)

axR.axhline(0, color="k", lw=0.8, ls="--")
axR.set_xlabel("R [Bohr]")
axR.set_ylabel(r"$\Delta$E = E(A) + E(B) - E(AB) [a.u.]")
axR.set_title("Binding energy")
axR.grid(alpha=0.3)
axR.legend(fontsize=8)

fig.suptitle(Path(args.csv).stem)
fig.tight_layout()

out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")
fig.savefig(out, dpi=150)
fig.savefig(out.with_suffix(".svg"))
print("saved →", out)
print("saved →", out.with_suffix(".svg"))
