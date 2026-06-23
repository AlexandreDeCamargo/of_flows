"""
PES (left) and binding-energy curve (right), side by side — matplotlib.

  MC          (line + o)  : per bond length, mean of training_metrics_ema.csv over an
                            epoch range (E + CC) + E_NN  ->  E_total.
  Integration (pentagons) : E_total from energy_summary.json (quadrature_scan.py).

Right panel: binding energy [a.u.] = E_total(R) - Σ_atoms E(atom), where the atomic
references come from atom_energies.csv (column E_ema_Ha for the MC line, E_grid_Ha for
the integration points), summed over the molecule's constituent atoms.

Usage
-----
python plot_pes_mpl.py --H2
python plot_pes_mpl.py --N2 --epoch_min 9000 --epoch_max 10000
python plot_pes_mpl.py --H2 --atom_csv Results/atom_energies.csv
python plot_pes_mpl.py --H2 --results_root /scratch/al3x/MyRuns --out /tmp/h2.png

Output: static PNG + SVG under Results/{mol}/ (no HTML, no extra dependencies).
Run quadrature_scan.py first so each bl_* has an energy_summary.json.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")              # headless: works on the cluster, no display needed
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
KNOWN_MOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
              "H2", "N2", "O2", "F2", "HF", "CO", "LiH", "H10"]
Z_TABLE = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
           "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    allow_abbrev=False)  # so --H is not a prefix of --H2 / --H10
for _m in KNOWN_MOLS:
    parser.add_argument(f"--{_m}", action="store_true", help=f"Plot molecule {_m}")
parser.add_argument("--mol", nargs="+", default=[], metavar="NAME",
                    help="Molecule name(s) (alternative to the flags)")
parser.add_argument("--method", default=None,
                    help="Restrict to one method directory (default: all methods)")
parser.add_argument("--epoch_min", type=int, default=None,
                    help="Average EMA rows with epoch >= this")
parser.add_argument("--epoch_max", type=int, default=None,
                    help="Average EMA rows with epoch <= this")
parser.add_argument("--window", type=int, default=1000,
                    help="If no epoch range given: average the last N EMA rows")
parser.add_argument("--results_root", default=None,
                    help="Override Results root (default: <script_dir>/Results)")
parser.add_argument("--atom_csv", default=None,
                    help="Path to atom_energies.csv (default: search mol dir / root / CWD)")
parser.add_argument("--out", default=None,
                    help="Output image path (default: Results/{mol}/pes_{mol}.png)")
args = parser.parse_args()

selected = list(args.mol) + [m for m in KNOWN_MOLS if getattr(args, m)]
selected = list(dict.fromkeys(selected))
if not selected:
    parser.error("Pick a molecule, e.g. --H2  or  --mol H2")

root = Path(args.results_root).resolve() if args.results_root else (_SCRIPT_DIR / "Results")


# ── helpers ───────────────────────────────────────────────────────────────────
def ema_mean(df: pd.DataFrame):
    """Electronic energy: mean of (E + CC) over the chosen epoch range / window."""
    if args.epoch_min is not None or args.epoch_max is not None:
        lo = args.epoch_min if args.epoch_min is not None else df["epoch"].min()
        hi = args.epoch_max if args.epoch_max is not None else df["epoch"].max()
        sel = df[(df["epoch"] >= lo) & (df["epoch"] <= hi)]
    else:
        sel = df.tail(args.window)
    if sel.empty:
        return None
    e = float(sel["E"].mean())
    if "CC" in sel.columns:
        e += float(sel["CC"].mean())
    return e


def enn_fallback(mol: str, R: float):
    """E_NN for simple geometries — used only if energy_summary.json is missing."""
    m = re.fullmatch(r"([A-Z][a-z]?)2", mol)               # homonuclear diatomic
    if m and m.group(1) in Z_TABLE:
        z = Z_TABLE[m.group(1)]
        return z * z / R
    pairs = {"HF": ("H", "F"), "CO": ("C", "O"), "LiH": ("Li", "H")}
    if mol in pairs:
        a, b = pairs[mol]
        return Z_TABLE[a] * Z_TABLE[b] / R
    m = re.fullmatch(r"H(\d+)", mol)                        # linear equal-spaced Hn chain
    if m:
        n = int(m.group(1))
        return sum(1.0 / (abs(i - j) * R) for i in range(n) for j in range(i + 1, n))
    return None


def constituents(mol: str) -> dict:
    """{element: count} from a formula, e.g. H2->{H:2}, HF->{H:1,F:1}, H10->{H:10}."""
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", mol):
        if el:
            out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def load_atom_csv(mol_dir: Path):
    """Find and load atom_energies.csv (indexed by atom symbol)."""
    cands = []
    if args.atom_csv:
        cands.append(Path(args.atom_csv))
    cands += [mol_dir / "atom_energies.csv",
              root / "atom_energies.csv",
              Path("atom_energies.csv")]
    for p in cands:
        if p.exists():
            return pd.read_csv(p).set_index("atom")
    return None


def atom_ref(atom_df, mol: str, col: str):
    """Σ_atoms count * E(atom) from the given column; None if any atom is missing."""
    if atom_df is None:
        return None
    tot = 0.0
    for el, n in constituents(mol).items():
        if el not in atom_df.index:
            return None
        tot += n * float(atom_df.loc[el, col])
    return tot


def gather(method_dir: Path, mol: str) -> pd.DataFrame:
    rows = []
    for bl in sorted(method_dir.glob("bl_*"), key=lambda d: float(d.name.split("_")[1])):
        R = float(bl.name.split("_")[1])
        grid_E = e_nn = epoch = None

        es = bl / "energy_summary.json"
        if es.exists():
            with open(es) as f:
                d = json.load(f)
            grid_E = d.get("E_total")
            e_nn   = d.get("E_NN")
            epoch  = d.get("epoch")
        if e_nn is None:
            e_nn = enn_fallback(mol, R)

        ema_E = None
        ec = bl / "training_metrics_ema.csv"
        if ec.exists() and e_nn is not None:
            try:
                df = pd.read_csv(ec)
            except Exception:
                df = None
            if df is not None and not df.empty and "E" in df.columns:
                em = ema_mean(df)
                if em is not None:
                    ema_E = em + e_nn

        rows.append(dict(R=R, grid_E=grid_E, ema_E=ema_E, epoch=epoch))
    return pd.DataFrame(rows).sort_values("R").reset_index(drop=True)


def short(method: str) -> str:
    """Readable legend label: strip the parts common to every run."""
    return (method.replace("tf_w_lam0.2_", "")
                  .replace("_dopri8", "")
                  .replace("_sched_MIX", ""))


def plot_mol(mol: str):
    mdir = root / mol
    if not mdir.is_dir():
        print(f"[{mol}] not found at {mdir}")
        return

    if args.method:
        methods = [mdir / args.method]
    else:
        methods = sorted(d for d in mdir.iterdir() if d.is_dir())

    # Atomic references for the binding-energy panel (same CSV for every method).
    atom_df  = load_atom_csv(mdir)
    ref_ema  = atom_ref(atom_df, mol, "E_ema_Ha")
    ref_grid = atom_ref(atom_df, mol, "E_grid_Ha")
    if ref_ema is None and ref_grid is None:
        print(f"[{mol}] atom_energies.csv not found / missing atoms — binding panel empty")

    colors = plt.cm.tab10.colors
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
    any_data = False
    bind_rows = []

    for i, md in enumerate(methods):
        if not md.is_dir():
            print(f"[{mol}] method dir not found: {md.name}")
            continue
        df = gather(md, mol)
        if df.empty:
            continue
        c = colors[i % len(colors)]

        ema  = df.dropna(subset=["ema_E"])
        grid = df.dropna(subset=["grid_E"])

        # ── left: PES ─────────────────────────────────────────────────────────
        if not ema.empty:
            any_data = True
            axL.plot(ema["R"], ema["ema_E"], "-o", color=c, lw=2, ms=12, label="MC")
        if not grid.empty:
            any_data = True
            axL.scatter(grid["R"], grid["grid_E"], color="orange", marker="p",
                        s=100, edgecolors="none", linewidths=0.6, zorder=5,
                        label="Integration")

        # ── right: ΔE = E(A) + E(B) - E(AB) = Σ E(atom) - E_total  (>0 = bound)
        if not ema.empty and ref_ema is not None:
            axR.plot(ema["R"], ref_ema - ema["ema_E"], "-o", color=c, lw=2, ms=12,
                     label="MC")
        if not grid.empty and ref_grid is not None:
            axR.scatter(grid["R"], ref_grid - grid["grid_E"], color="orange", marker="p",
                        s=100, edgecolors="none", linewidths=0.6, zorder=5,
                        label="Integration")

        # binding values:  ΔE = E(A) + E(B) - E(AB) = Σ E(atom) - E_total  (>0 = bound)
        for _, r in df.iterrows():
            row = {"method": md.name, "R_bohr": float(r["R"]), "epoch": r["epoch"]}
            if pd.notna(r["ema_E"]) and ref_ema is not None:
                row["E_atoms_mc"] = ref_ema
                row["E_AB_mc"]    = float(r["ema_E"])
                row["dE_mc_Ha"]   = ref_ema - float(r["ema_E"])
            if pd.notna(r["grid_E"]) and ref_grid is not None:
                row["E_atoms_grid"] = ref_grid
                row["E_AB_grid"]    = float(r["grid_E"])
                row["dE_grid_Ha"]   = ref_grid - float(r["grid_E"])
            bind_rows.append(row)

    if not any_data:
        print(f"[{mol}] no data — run quadrature_scan.py first "
              f"(creates energy_summary.json in each bl_*).")
        plt.close(fig)
        return

    axL.set_xlabel("R [Bohr]")
    axL.set_ylabel(r"E[$\rho$] + V$_{NN}$(R) [a.u.]")
    axL.legend(fontsize=9)

    axR.axhline(0, color="k", lw=0.8, ls="--")
    axR.set_xlabel("R [Bohr]")
    axR.set_ylabel(r"$\Delta$E = E(A) + E(B) - E(AB) [a.u.]")
    axR.legend(fontsize=9)

    fig.tight_layout()

    out = Path(args.out).resolve() if args.out else mdir / f"pes_{mol}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"[{mol}] saved → {out}")
    print(f"[{mol}] saved → {out.with_suffix('.svg')}")

    if bind_rows:
        bdf = (pd.DataFrame(bind_rows)
                 .sort_values(["method", "R_bohr"]).reset_index(drop=True))
        bcsv = out.with_name(f"binding_{mol}.csv")
        bdf.to_csv(bcsv, index=False, float_format="%.8f")
        print(f"[{mol}] saved → {bcsv}")


for mol in selected:
    plot_mol(mol)
