"""
Grid-quadrature total energy for every bond length of a molecule.

Thin CLI around ``of_flows/quadrature.py``: it walks every method directory and
every bl_* subdirectory under Results/{mol}/, calls
``quadrature.grid_energy_from_checkpoint`` on each (which builds the PySCF grid,
evaluates ρ_φ via the flow, and integrates all energy terms), and writes one CSV
per molecule.  For molecules it also integrates the constituent single atoms
under the same method tag and reports the binding energy.

Directory layout assumed (same as main.py):
    Results/{mol}/{method}/bl_X.XXXX/
        Checkpoints/checkpoint_*.eqx
        job_params.json
    Results/{atom}/{method}/bl_0.0000/   (binding reference)
Results/ is located next to this script, so it runs from anywhere.

Usage
-----
python quadrature_scan.py --H2
python quadrature_scan.py --H2 --N2
python quadrature_scan.py --mol H2 H10
python quadrature_scan.py --H10 --recompute

Output (one CSV per molecule, under Results/{mol}/):
    Results/{mol}/quadrature_{mol}.csv
      columns: method, R_bohr, epoch, E_total, T, V_N, V_H, E_X, E_C, E_CC,
               E_NN, Ne_int, E_atoms, dE_bind_Ha
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import re
import argparse
from pathlib import Path

import jax
import pandas as pd

from quadrature import grid_energy_from_checkpoint

_SCRIPT_DIR = Path(__file__).resolve().parent

KNOWN_MOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
              "H2", "N2", "O2", "F2", "HF", "CO", "LiH", "H10"]
SINGLE_ATOMS = {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    allow_abbrev=False)  # so --H is not treated as a prefix of --H2 / --H10
for _m in KNOWN_MOLS:
    parser.add_argument(f"--{_m}", action="store_true", help=f"Scan molecule {_m}")
parser.add_argument("--mol", type=str, nargs="+", default=[], metavar="NAME",
                    help="Molecule name(s) to scan (alternative to the flags)")
parser.add_argument("--results_root", type=str, default=None,
                    help="Override Results root (default: <script_dir>/Results)")
parser.add_argument("--bs", type=int, default=256, help="Grid chunk size")
parser.add_argument("--grid_level", type=int, default=3, help="PySCF grid level")
parser.add_argument("--recompute", action="store_true",
                    help="Re-run grid integration even if energy_summary.json is cached")
parser.add_argument("--out", type=str, default=None,
                    help="Output CSV path (default: Results/{mol}/quadrature_{mol}.csv)")
args = parser.parse_args()

selected = list(args.mol) + [m for m in KNOWN_MOLS if getattr(args, m)]
selected = list(dict.fromkeys(selected))
if not selected:
    parser.error("No molecule selected. Use a flag (e.g. --H2) or --mol H2 [...].")

root = Path(args.results_root).resolve() if args.results_root else (_SCRIPT_DIR / "Results")
print(f"Results root : {root}\n")


def constituents(mol: str) -> dict:
    """{element: count} from a formula, e.g. N2->{N:2}, HF->{H:1,F:1}, H10->{H:10}."""
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", mol):
        if el:
            out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def atom_reference(method_name: str, mol: str):
    """Grid energy reference Σ_atoms count·E(atom) under the same method tag.
    Returns (E_atoms, {element: E_atom}) or (None, None) if any atom is missing."""
    total = 0.0
    detail = {}
    for el, n in constituents(mol).items():
        adir = root / el / method_name / "bl_0.0000"
        if not (adir / "job_params.json").exists():
            print(f"  atom reference: {el} not found at {adir} — binding skipped")
            return None, None
        try:
            data = grid_energy_from_checkpoint(
                adir, grid_level=args.grid_level, chunk=args.bs, recompute=args.recompute)
        except Exception as e:
            print(f"  atom reference: {el} FAILED — {e}")
            return None, None
        detail[el] = data['E_total']
        total += n * data['E_total']
    return total, detail


def scan_molecule(mol: str):
    mol_dir = root / mol
    if not mol_dir.is_dir():
        print(f"[{mol}] SKIP — {mol_dir} not found\n")
        return

    is_atom = mol in SINGLE_ATOMS
    rows = []
    for method_dir in sorted(d for d in mol_dir.iterdir() if d.is_dir()):
        bl_dirs = sorted(method_dir.glob("bl_*"),
                         key=lambda d: float(d.name.split("_")[1]))
        if not bl_dirs:
            continue
        print(f"[{mol}] method: {method_dir.name}  ({len(bl_dirs)} bond lengths)")

        # Single-atom reference for the binding energy (same method tag).
        E_atoms = None
        if not is_atom:
            E_atoms, detail = atom_reference(method_dir.name, mol)
            if E_atoms is not None:
                ref = "  ".join(f"{n}*E({el})={detail[el]:+.6f}"
                                for el, n in constituents(mol).items())
                print(f"  atom reference (grid): {ref}  ->  Σ = {E_atoms:+.6f} Ha")

        for bl_dir in bl_dirs:
            if not (bl_dir / "job_params.json").exists():
                print(f"  {bl_dir.name}: missing job_params.json — skipping")
                continue
            try:
                data = grid_energy_from_checkpoint(
                    bl_dir, grid_level=args.grid_level, chunk=args.bs,
                    recompute=args.recompute)
            except Exception as e:
                print(f"  {bl_dir.name}: FAILED — {e}")
                continue
            row = {
                "method":  method_dir.name,
                "R_bohr":  data['bond_length'],
                "epoch":   data.get('epoch', '?'),
                "E_total": data['E_total'],
                "T":       data['T'],
                "V_N":     data['V_N'],
                "V_H":     data['V_H'],
                "E_X":     data['E_X'],
                "E_C":     data.get('E_C', 0.0),
                "E_CC":    data.get('E_CC', 0.0),
                "E_NN":    data['E_NN'],
                "Ne_int":  data['Ne_integral'],
            }
            if E_atoms is not None:
                row["E_atoms"]    = E_atoms
                row["dE_bind_Ha"] = E_atoms - data['E_total']   # ΔE = ΣE(atom) - E(mol)
            rows.append(row)
            msg = (f"    R={data['bond_length']:.4f} Bohr  epoch={data.get('epoch','?'):>6}"
                   f"  E_total={data['E_total']:+.6f} Ha")
            if E_atoms is not None:
                msg += f"  ΔE={E_atoms - data['E_total']:+.6f} Ha"
            print(msg)
            jax.clear_caches()
            gc.collect()
        print()

    if not rows:
        print(f"[{mol}] nothing to write (no checkpoints found)\n")
        return

    df = (pd.DataFrame(rows)
            .sort_values(["method", "R_bohr"])
            .reset_index(drop=True))
    out_path = (Path(args.out).resolve() if args.out
                else mol_dir / f"quadrature_{mol}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.8f")

    print("=" * 96)
    print(df.to_string(index=False))
    print("=" * 96)
    print(f"[{mol}] saved → {out_path}\n")


for mol in selected:
    scan_molecule(mol)
