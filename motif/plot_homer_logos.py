#!/usr/bin/env python3
"""Render HOMER *.motif PWMs to PNG logos (website display colors)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Same palette as create_motif_logo.ipynb (website / paper figures)
COLOR_SCHEME = {
    "A": "#65a455",
    "C": "#2e45a4",
    "G": "#fda562",
    "U": "#d54f3f",
}


def parse_homer_motif(path: Path) -> list[tuple[str, np.ndarray]]:
    """Parse HOMER .motif file → [(label, Lx4 ACGU matrix), ...]."""
    text = path.read_text(encoding="utf-8", errors="replace")
    motifs: list[tuple[str, np.ndarray]] = []
    blocks = [b.strip() for b in text.split(">") if b.strip()]
    for bi, block in enumerate(blocks):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        header = lines[0].split("\t")
        label = header[1] if len(header) > 1 else f"{path.stem}_{bi + 1}"
        rows: list[list[float]] = []
        for ln in lines[1:]:
            parts = re.split(r"\s+", ln)
            if len(parts) != 4:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
        if rows:
            motifs.append((label, np.asarray(rows, dtype=np.float64)))
    return motifs


def plot_logo(matrix: np.ndarray, output_path: Path, dpi: int = 200) -> None:
    df = pd.DataFrame(matrix, columns=["A", "C", "G", "U"])
    df = df.div(df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.25)
    fig, ax = plt.subplots(figsize=(4, 2))
    logomaker.Logo(
        df,
        ax=ax,
        color_scheme=COLOR_SCHEME,
        baseline_width=0,
        show_spines=False,
    )
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format="png")
    plt.close(fig)


def render_homer_dir(homer_dir: Path, logo_dir: Path, dpi: int = 200) -> list[Path]:
    out: list[Path] = []
    logo_dir.mkdir(parents=True, exist_ok=True)
    for motif_path in sorted(homer_dir.glob("*.motif")):
        for i, (label, mat) in enumerate(parse_homer_motif(motif_path), start=1):
            safe = re.sub(r"[^\w.\-]+", "_", label)
            png = logo_dir / f"{motif_path.stem}_logo{i}_{safe}.png"
            plot_logo(mat, png, dpi=dpi)
            out.append(png)
            print(f"saved {png.name}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="HOMER .motif → PNG logos")
    ap.add_argument("homer_dir", type=Path, help="Directory of *.motif files")
    ap.add_argument("logo_dir", type=Path, help="Output directory for PNGs")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    paths = render_homer_dir(args.homer_dir, args.logo_dir, dpi=args.dpi)
    print(f"wrote {len(paths)} logos -> {args.logo_dir}")


if __name__ == "__main__":
    main()
