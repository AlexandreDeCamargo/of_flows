"""
Quick PES plot from EMA training logs — no grid integration needed.

Reads training_metrics_ema.csv from each bl_* directory and uses the
last epoch's EMA energy (E + CC) as E_total.

Usage
-----
# PES only:
python plot_pes_ema.py \
    --scan_dir Results/H2/tf_w_lam0.2_hutcheon_lda_none_dopri8_promolecular_sched_MIX

# With binding energy (needs H atom dir):
python plot_pes_ema.py \
    --scan_dir Results/N2/tf_w_lam0.2_none_lda_none_dopri8_promolecular_sched_MIX_hart_COULOMB_ALLPAIRS \
    --atom_dir Results/N/tf_w_lam0.2_none_lda_none_dopri8_promolecular_sched_MIX_hart_COULOMB_ALLPAIRS/bl_0.0000

# Read R=8.0 and R=9.0 at epoch 20000, the rest at their last epoch:
python plot_pes_ema.py --scan_dir Results/N2/... --epoch_at 8.0:20000 9.0:20000
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--scan_dir", type=str, required=True,
                    help="Method directory containing bl_X.XXXX subdirectories")
parser.add_argument("--atom_dir", type=str, default=None,
                    help="bl_0.0000 directory for the H atom (binding energy reference)")
parser.add_argument("--pes_csv", type=str, default=None,
                    help="pes.csv from scan_pes.py to overlay as grid-integration "
                         "points. If omitted, looks for pes.csv inside --scan_dir.")
parser.add_argument("--avg_window", type=int, default=1,
                    help="Average the last N rows of training_metrics_ema.csv "
                         "instead of taking just the last value (default: 500).")
parser.add_argument("--bls", type=float, nargs="+", default=None,
                    help="Only include these bond lengths, e.g. --bls 2.0 3.0 4.0 9.0. "
                         "If omitted, include all bl_* directories found.")
parser.add_argument("--epoch_at", type=str, nargs="+", default=None, metavar="R:EPOCH",
                    help="Per-bond-length epoch override, e.g. --epoch_at 8.0:20000 9.0:20000. "
                         "Those bond lengths use the EMA as of that epoch; all others use "
                         "their last epoch.")
args = parser.parse_args()

scan_dir = Path(args.scan_dir).resolve()

# Parse --epoch_at "R:EPOCH" pairs into {round(R,4): epoch}
EPOCH_OVERRIDE = {}
if args.epoch_at:
    for pair in args.epoch_at:
        if ":" not in pair:
            parser.error(f"--epoch_at expects R:EPOCH pairs, got '{pair}'")
        r_str, e_str = pair.split(":", 1)
        EPOCH_OVERRIDE[round(float(r_str), 4)] = int(e_str)
    print(f"Epoch overrides: {EPOCH_OVERRIDE}")


def read_last_ema(bl_dir: Path, window: int = 500, at_epoch: int = None):
    """Return (E_electronic, epoch) averaged over the last `window` rows of
    training_metrics_ema.csv.  Epoch returned is the final one (window is just
    smoothing the EMA noise).
    E_electronic = E + CC  (does NOT include nuclear repulsion E_NN).
    If `at_epoch` is given, the log is first truncated to rows with
    epoch <= at_epoch, so the value is read *as of* that epoch.
    """
    csv = bl_dir / "training_metrics_ema.csv"
    if not csv.exists():
        return None, None
    df = pd.read_csv(csv)
    if df.empty:
        return None, None
    if at_epoch is not None:
        df = df[df["epoch"] <= at_epoch]      # read the EMA as of this epoch
        if df.empty:
            return None, None
    tail = df.tail(window)
    E_elec = float(tail["E"].mean())
    if "CC" in tail.columns:
        E_elec += float(tail["CC"].mean())
    epoch = int(df.iloc[-1]["epoch"])
    return E_elec, epoch


def e_nn(mol_name: str, R: float) -> float:
    """Nuclear-nuclear repulsion energy [Ha] for a diatomic at bond length R [Bohr]."""
    Z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6,
         "N": 7, "O": 8, "F": 9, "Ne": 10}
    # homonuclear diatomics: mol_name = element symbol × 2 (e.g. "H2", "N2")
    elem = mol_name.rstrip("0123456789")
    if mol_name in ("HF",):
        za, zb = Z["H"], Z["F"]
    elif mol_name == "CO":
        za, zb = Z["C"], Z["O"]
    elif mol_name == "NO":
        za, zb = Z["N"], Z["O"]
    else:
        za = zb = Z.get(elem, 1)
    return za * zb / R if R > 0 else 0.0


# ── H atom reference ──────────────────────────────────────────────────────────
E_atom = None
if args.atom_dir is not None:
    E_atom, ep_atom = read_last_ema(Path(args.atom_dir).resolve(), window=args.avg_window)
    if E_atom is not None:
        print(f"E(atom) = {E_atom:+.6f} Ha  (epoch {ep_atom})")
    else:
        print(f"WARNING: could not read atom EMA from {args.atom_dir}")
        E_atom = None

# ── Scan over bl_* directories ────────────────────────────────────────────────
bl_dirs = sorted(scan_dir.glob("bl_*"),
                 key=lambda d: float(d.name.split("_")[1]))

if not bl_dirs:
    raise FileNotFoundError(f"No bl_* directories found in {scan_dir}")

if args.bls is not None:
    keep = {round(bl, 4) for bl in args.bls}
    bl_dirs = [d for d in bl_dirs if round(float(d.name.split("_")[1]), 4) in keep]
    if not bl_dirs:
        raise FileNotFoundError(
            f"None of --bls {args.bls} match any bl_X.XXXX in {scan_dir}")
    print(f"Filtered to {len(bl_dirs)} requested bond lengths: "
          f"{[d.name for d in bl_dirs]}")

import json, re

# detect molecule name from first bl_* job_params.json
mol_name = "H2"
for d in bl_dirs:
    jp = d / "job_params.json"
    if jp.exists():
        mol_name = json.load(open(jp))["mol_name"]
        break
print(f"Molecule: {mol_name}")

# Parse mol_name for LaTeX labels: A_n -> atom_sym='A', mol_latex='\mathrm{A}_n'
_m = re.fullmatch(r"([A-Z][a-z]?)(\d+)?", mol_name)
if _m and _m.group(2):
    atom_sym = _m.group(1)
    mol_latex = rf"\mathrm{{{atom_sym}}}_{{{_m.group(2)}}}"
else:
    atom_sym  = mol_name
    mol_latex = rf"\mathrm{{{mol_name}}}"
    if mol_name in ("HF", "CO", "NO"):
        print(f"WARNING: {mol_name} is heteronuclear — binding uses 2*E_atom, "
              f"which assumes homonuclear dissociation and will be physically wrong")

rows = []
for bl_dir in bl_dirs:
    R = float(bl_dir.name.split("_")[1])
    if R == 0.0:
        continue                      # skip atom directory if present
    at = EPOCH_OVERRIDE.get(round(R, 4))            # None unless this R is overridden
    E_elec, epoch = read_last_ema(bl_dir, window=args.avg_window, at_epoch=at)
    if E_elec is None:
        print(f"  bl={R:.4f}: no EMA csv — skipping")
        continue
    E_NN   = e_nn(mol_name, R)
    E_total = E_elec + E_NN           # add nuclear repulsion
    row = {"R": R, "E_total": E_total, "epoch": epoch}
    if E_atom is not None:
        row["bind_Ha"] = E_total - 2.0 * E_atom   # E(H2) - 2E(H)
    rows.append(row)
    tag = f"  R={R:.4f}  epoch={epoch:>6}  E={E_total:+.6f} Ha  (E_NN={E_NN:+.4f})"
    if E_atom is not None:
        tag += f"  bind={row['bind_Ha']:+.6f} Ha"
    print(tag)

if not rows:
    raise RuntimeError("No data found — check scan_dir.")

df = pd.DataFrame(rows).sort_values("R").reset_index(drop=True)
max_epoch = df["epoch"].max()
complete  = df["epoch"] == max_epoch

# ── Optional: grid-integration results from scan_pes.py ──────────────────────
pes_path = Path(args.pes_csv).resolve() if args.pes_csv else scan_dir / "pes.csv"
pes_df   = None
if pes_path.exists():
    pes_df = pd.read_csv(pes_path).sort_values("R_bohr").reset_index(drop=True)
    if args.bls is not None:
        keep = {round(bl, 4) for bl in args.bls}
        pes_df = pes_df[pes_df["R_bohr"].round(4).isin(keep)].reset_index(drop=True)
    print(f"\nGrid overlay: {pes_path}  ({len(pes_df)} points)")
else:
    print(f"\nNo pes.csv at {pes_path} — plotting EMA only")

# ── Side-by-side binding-energy comparison (EMA vs grid) ─────────────────────
if pes_df is not None and E_atom is not None and "E_bind_Ha" in pes_df.columns:
    cmp = df[["R", "bind_Ha"]].merge(
        pes_df[["R_bohr", "E_bind_Ha"]].rename(columns={"R_bohr": "R",
                                                        "E_bind_Ha": "grid_bind_Ha"}),
        on="R", how="outer").sort_values("R").reset_index(drop=True)
    cmp["delta_Ha"] = cmp["bind_Ha"] - cmp["grid_bind_Ha"]
    print("\n=== Binding energy: EMA vs grid (Ha) ===")
    print(f"{'R':>8}  {'EMA':>12}  {'Grid':>12}  {'Δ(EMA-Grid)':>14}")
    for _, r in cmp.iterrows():
        ema_s  = f"{r['bind_Ha']:+12.6f}"      if pd.notna(r['bind_Ha'])      else f"{'—':>12}"
        grid_s = f"{r['grid_bind_Ha']:+12.6f}" if pd.notna(r['grid_bind_Ha']) else f"{'—':>12}"
        d_s    = f"{r['delta_Ha']:+14.6f}"     if pd.notna(r['delta_Ha'])     else f"{'—':>14}"
        print(f"{r['R']:8.4f}  {ema_s}  {grid_s}  {d_s}")

# ── Plot ──────────────────────────────────────────────────────────────────────
n_panels = 2 if E_atom is not None else 1
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels + 1, 4))
if n_panels == 1:
    axes = [axes]

R_vals = df["R"].values

# Panel 1: PES (E_total)
ax = axes[0]
ax.plot(R_vals[complete],  df["E_total"].values[complete],
        "o-", color="tab:blue", lw=1.8, label=f"EMA  (epoch={max_epoch})")
ax.plot(R_vals[~complete], df["E_total"].values[~complete],
        "^", color="tab:blue", alpha=0.5, markerfacecolor="none",
        markersize=7, label="incomplete")
if pes_df is not None:
    ax.plot(pes_df["R_bohr"], pes_df["E_total"], "o", color="gold",
            markersize=6, markeredgecolor="black", markeredgewidth=0.4,
            linestyle="none", label="scan_pes (grid)", zorder=5)
ax.set_xlabel("R [Bohr]")
ax.set_ylabel("E [Ha]")
ax.set_title("Potential Energy Surface")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

# Panel 2: binding energy  E(mol) - 2 E(atom)
if E_atom is not None:
    ax2 = axes[1]
    ax2.plot(R_vals[complete],  df["bind_Ha"].values[complete],
             "o-", color="tab:orange", lw=1.8, label=f"EMA  (epoch={max_epoch})")
    ax2.plot(R_vals[~complete], df["bind_Ha"].values[~complete],
             "^", color="tab:orange", alpha=0.5, markerfacecolor="none", markersize=7)
    if pes_df is not None and "E_bind_Ha" in pes_df.columns:
        # scan_pes already stores E_bind = E(mol) - 2E(atom) — plot directly
        ax2.plot(pes_df["R_bohr"], pes_df["E_bind_Ha"], "o", color="gold",
                 markersize=6, markeredgecolor="black", markeredgewidth=0.4,
                 linestyle="none", label="scan_pes (grid)", zorder=5)
    ax2.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel("R [Bohr]")
    ax2.set_ylabel(rf"$E({mol_latex}) - 2E(\mathrm{{{atom_sym}}})$  [Ha]")
    ax2.set_title("Binding Energy")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

fig.suptitle(scan_dir.name, fontsize=8)
fig.tight_layout()

out = scan_dir / "pes_ema.png"
fig.savefig(out, dpi=150)
fig.savefig(out.with_suffix(".svg"), transparent=True)
print(f"\nSaved → {out}")
