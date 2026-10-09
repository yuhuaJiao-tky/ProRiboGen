#!/usr/bin/env python3
"""One-shot motif pipeline: MEME → keep all significant motifs → paper-style logos.

For each *.fasta under the input folder, writes into <output>/<stem>/:
  meme.txt              MEME text output
  <stem>.motif          HOMER PWM(s), all motifs with E ≤ 0.05
  logo1.png … logoN.png paper-style logos, ranked by E (logo1 = most significant)

MEME's own EPS/PNG logos are removed so they do not mix with our style.
Needs `meme` on PATH and: logomaker, matplotlib, pandas, numpy.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_MOTIF_DIR = Path(__file__).resolve().parent
_DEFAULT_BFILE = _MOTIF_DIR / "background.txt"
_E_RE = re.compile(r"\bE=\s*([0-9.eE+-]+)")
_MATRIX_ROW_RE = re.compile(r"^([0-9.eE+-]+\s+){3}[0-9.eE+-]+$")
_COLOR_SCHEME = {
    "A": "#65a455",
    "C": "#2e45a4",
    "G": "#fda562",
    "U": "#d54f3f",
}
_MEME_LOGO_GLOBS = (
    "logo*.png",
    "logo*.eps",
    "logo*.pdf",
    "logo_*.png",
    "logo_*.eps",
)


def _require_plotting() -> None:
    try:
        import logomaker  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"missing plotting dependency: {e}\n"
            "In the meme_new env run:\n"
            "  python -m pip install logomaker matplotlib pandas numpy"
        ) from e


def parse_meme_motifs(meme_txt: Path) -> list[tuple[float, str, np.ndarray]]:
    """Return [(E, name, Lx4 matrix), ...] for every motif in meme.txt."""
    chunks: list[tuple[float, str, np.ndarray]] = []
    current_name: str | None = None
    current_e = float("inf")
    in_matrix = False
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal rows, current_name, current_e
        if current_name is not None and rows:
            chunks.append((current_e, current_name, np.asarray(rows, dtype=np.float64)))
        rows = []

    with meme_txt.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("MOTIF"):
                flush()
                parts = line.split()
                current_name = parts[1] if len(parts) > 1 else "?"
                current_e = float("inf")
                in_matrix = False
                continue
            if line.startswith("letter-probability matrix:"):
                m = _E_RE.search(line)
                if m:
                    try:
                        current_e = float(m.group(1))
                    except ValueError:
                        current_e = float("inf")
                in_matrix = True
                continue
            if in_matrix and line:
                if _MATRIX_ROW_RE.match(line):
                    rows.append([float(x) for x in line.split()])
                else:
                    in_matrix = False
    flush()
    chunks.sort(key=lambda x: (x[0], x[1]))
    return chunks


def write_homer_motif(
    out_path: Path,
    rbp_name: str,
    motifs: list[tuple[float, str, np.ndarray]],
    e_max: float = 0.05,
) -> int:
    """Write all motifs with E ≤ e_max into one HOMER .motif file."""
    kept = [(e, name, mat) for e, name, mat in motifs if e <= e_max]
    if not kept:
        return 0
    lines: list[str] = []
    for i, (_e, name, mat) in enumerate(kept, start=1):
        lines.append(f">{name}\t{rbp_name}_{i}\t+")
        for row in mat:
            lines.append("\t".join(f"{float(x):.6f}" for x in row))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(kept)


def plot_logo(matrix: np.ndarray, output_path: Path, dpi: int = 600) -> None:
    import logomaker
    import matplotlib.pyplot as plt

    df = pd.DataFrame(matrix, columns=["A", "C", "G", "U"])
    denom = df.sum(axis=1).replace(0, np.nan)
    df = df.div(denom, axis=0).fillna(0.25)
    fig, ax = plt.subplots(figsize=(4, 2))
    logomaker.Logo(
        df,
        ax=ax,
        color_scheme=_COLOR_SCHEME,
        baseline_width=0,
        show_spines=False,
    )
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format="png")
    plt.close(fig)


def clear_meme_native_logos(out_dir: Path) -> None:
    for pattern in _MEME_LOGO_GLOBS:
        for p in out_dir.glob(pattern):
            try:
                p.unlink()
            except OSError:
                pass


def run_meme(
    fasta_file: Path,
    output_dir: Path,
    bfile: Path,
    *,
    minw: int = 6,
    maxw: int = 8,
    nmotifs: int = 4,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "meme",
        str(fasta_file),
        "-oc",
        str(output_dir),
        "-rna",
        "-minw",
        str(minw),
        "-maxw",
        str(maxw),
        "-nmotifs",
        str(nmotifs),
        "-markov_order",
        "0",
        "-mod",
        "zoops",
        "-objfun",
        "classic",
        "-nostatus",
        "-bfile",
        str(bfile),
    ]
    subprocess.run(cmd, check=True)
    meme_txt = output_dir / "meme.txt"
    if not meme_txt.is_file():
        raise FileNotFoundError(f"MEME finished but no meme.txt in {output_dir}")
    clear_meme_native_logos(output_dir)
    return meme_txt


def process_one(
    fasta_file: Path,
    output_root: Path,
    bfile: Path,
    *,
    e_max: float,
    dpi: int,
    minw: int,
    maxw: int,
    nmotifs: int,
) -> str:
    stem = fasta_file.stem
    out_dir = output_root / stem
    meme_txt = run_meme(
        fasta_file,
        out_dir,
        bfile,
        minw=minw,
        maxw=maxw,
        nmotifs=nmotifs,
    )
    motifs = parse_meme_motifs(meme_txt)
    n_sig = write_homer_motif(out_dir / f"{stem}.motif", stem, motifs, e_max=e_max)
    for i, (_e, _name, mat) in enumerate(motifs, start=1):
        plot_logo(mat, out_dir / f"logo{i}.png", dpi=dpi)
    return (
        f"{stem}: meme.txt, {n_sig} significant HOMER motif(s) (E≤{e_max}), "
        f"{len(motifs)} logo(s) -> {out_dir}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MEME + HOMER (all E≤threshold) + paper-style logos in one step"
    )
    ap.add_argument("input_folder", type=Path, help="Folder of *.fasta")
    ap.add_argument("output_folder", type=Path, help="Output root (one subdir per FASTA)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bfile", type=Path, default=_DEFAULT_BFILE)
    ap.add_argument("--e-max", type=float, default=0.05, help="HOMER significance cutoff")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--minw", type=int, default=6)
    ap.add_argument("--maxw", type=int, default=8)
    ap.add_argument("--nmotifs", type=int, default=4)
    args = ap.parse_args()

    if shutil.which("meme") is None:
        raise SystemExit(
            "`meme` not found on PATH. Activate meme_new (conda env from motif/meme.yml)."
        )
    _require_plotting()

    bfile = args.bfile.resolve()
    if not bfile.is_file():
        raise SystemExit(f"background file missing: {bfile}")

    inp = args.input_folder.expanduser().resolve()
    out = args.output_folder.expanduser().resolve()
    if not inp.is_dir():
        raise SystemExit(f"FASTA folder missing: {inp}")
    fasta_files = sorted(inp.glob("*.fasta"))
    if not fasta_files:
        raise SystemExit(f"no *.fasta in {inp}")
    out.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as pool:
        futs = {
            pool.submit(
                process_one,
                fa,
                out,
                bfile,
                e_max=args.e_max,
                dpi=args.dpi,
                minw=args.minw,
                maxw=args.maxw,
                nmotifs=args.nmotifs,
            ): fa
            for fa in fasta_files
        }
        for fut in as_completed(futs):
            fa = futs[fut]
            try:
                print(fut.result())
                ok += 1
            except Exception as e:
                failed.append(f"{fa.name}: {e}")
                print(f"FAIL {fa.name}: {e}", file=sys.stderr)

    print(f"Done: {ok}/{len(fasta_files)} under {out}")
    if failed:
        print("Failures:", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
    if ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
