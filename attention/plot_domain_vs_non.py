#!/usr/bin/env python3
"""Paired boxplot: per-protein mean attention in annotated domain vs outside.

Reads profiles produced by analyze_rbd_cross_attention.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import wilcoxon

COLOR_DOM = "#009E73"
COLOR_NON = "#0072B2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-dir", type=Path, required=True, help="analyze_rbd_cross_attention.py --out-dir")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    attn = args.attn_dir
    font = Path(__file__).resolve().parent.parent / "Arial.ttf"
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})

    idx = pd.read_csv(attn / "protein_index.csv")
    rows = []
    for _, r in idx.iterrows():
        csv_path = Path(str(r.profile_csv))
        if not csv_path.is_file():
            csv_path = attn / "profiles" / f"{r.p_id}_attention_mean.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if "in_annotated_domain" not in df.columns:
            continue
        rbd = df.loc[df.in_annotated_domain == 1, "attention_mean"].to_numpy(float)
        non = df.loc[df.in_annotated_domain == 0, "attention_mean"].to_numpy(float)
        if rbd.size == 0 or non.size == 0:
            continue
        rows.append({"p_id": r.p_id, "mean_domain": float(np.mean(rbd)), "mean_non": float(np.mean(non))})
    m = pd.DataFrame(rows)
    if m.empty:
        raise SystemExit(f"No annotated profiles under {attn}")
    m.to_csv(attn / "per_protein_mean_attn_domain_vs_non.csv", index=False)
    w = wilcoxon(m.mean_domain, m.mean_non, alternative="greater")

    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    bp = ax.boxplot(
        [m.mean_non.values, m.mean_domain.values],
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.0),
        boxprops=dict(linewidth=0.8, color="black"),
    )
    bp["boxes"][0].set_facecolor(COLOR_NON)
    bp["boxes"][1].set_facecolor(COLOR_DOM)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_alpha(0.75)
    ax.set_xticks([1, 2], ["non-domain", "domain"])
    ax.set_ylabel("Mean cross-attention")
    ax.set_title(f"n={len(m)}  Wilcoxon p={w.pvalue:.3g}", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = args.out or (attn / "Fig_attn_domain_vs_non.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    print(f"n={len(m)}  domain>non={(m.mean_domain > m.mean_non).sum()}  p={w.pvalue:.4g}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
