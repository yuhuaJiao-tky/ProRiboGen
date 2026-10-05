#!/usr/bin/env python3
"""RBD cross-attention analysis (domain vs non-domain)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import DataLoader, Dataset

PILOT_ROOT = Path(__file__).resolve().parent
GEN_ROOT = Path(
    os.environ.get(
        "GEN_ROOT",
        str(PILOT_ROOT.parent / "generation"),
    )
).resolve()
sys.path.insert(0, str(GEN_ROOT))

from sample import (
    ProteinEmbeddingStore,
    build_model_from_config,
    build_tokenizer_runtime,
    resolve_path,
)
from src.utils import (
    build_xt_and_labels,
    load_config,
    resume_model,
    rna_rbp_collate_fn,
)


def load_probe_rnas(csv_path: Path, p_id: str, n: int, seed: int, min_rna_len: int = 12) -> list[str]:
    df = pd.read_csv(csv_path, usecols=["p_id", "rna"], low_memory=False)
    target = str(p_id).strip().casefold()
    sub = df[df["p_id"].astype(str).str.strip().str.casefold() == target].copy()
    if sub.empty:
        raise KeyError(f"p_id not found in {csv_path}: {p_id}")
    sub["rna"] = sub["rna"].astype(str).str.upper().str.replace("T", "U")
    sub = sub[sub["rna"].str.len() >= min_rna_len]
    if sub.empty:
        raise ValueError(f"No RNAs >= {min_rna_len}bp for {p_id}")
    rng = np.random.default_rng(seed)
    take = min(n, len(sub))
    idx = rng.choice(len(sub), size=take, replace=False)
    return [str(sub.iloc[i]["rna"]) for i in idx]


class ProbeDataset(Dataset):
    def __init__(self, rnas: list[str], p_id: str) -> None:
        self.rnas = rnas
        self.p_id = p_id

    def __len__(self) -> int:
        return len(self.rnas)

    def __getitem__(self, idx: int) -> dict:
        rna = self.rnas[idx]
        return {"r_id": f"probe_{idx}", "rna": rna, "p_id": self.p_id, "rna_len": len(rna)}


@torch.no_grad()
def collect_cross_attention(
    model,
    *,
    input_ids: torch.Tensor,
    rna_attention_mask: torch.Tensor,
    protein_cond: torch.Tensor,
    protein_attention_mask: torch.Tensor,
) -> np.ndarray:
    """Mean protein attention per layer: [num_layers, L_protein]."""
    esm = model.esm
    input_shape = input_ids.size()
    extended_attention_mask = esm.get_extended_attention_mask(
        rna_attention_mask, input_shape
    )
    embedding_output = esm.embeddings(
        input_ids=input_ids,
        attention_mask=rna_attention_mask,
    )

    captured: list[torch.Tensor] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(storage: list[torch.Tensor]):
        def _hook(_module, _inputs, output):
            if len(output) > 1 and output[1] is not None:
                storage.append(output[1].detach())

        return _hook

    for layer in esm.encoder.layer:
        if hasattr(layer, "protein_conditioning_attention"):
            handles.append(
                layer.protein_conditioning_attention.register_forward_hook(make_hook(captured))
            )

    hidden_states = embedding_output
    for layer in esm.encoder.layer:
        layer_outputs = layer(
            hidden_states=hidden_states,
            protein_cond=protein_cond,
            attention_mask=extended_attention_mask,
            protein_attention_mask=protein_attention_mask,
            output_attentions=True,
        )
        hidden_states = layer_outputs[0]

    for h in handles:
        h.remove()

    if not captured:
        raise RuntimeError("No protein cross-attention tensors captured.")

    stacked = torch.stack(captured, dim=0)  # [layer, B, heads, L_rna, L_protein]
    bsz = rna_attention_mask.size(0)
    rna_mask = (
        rna_attention_mask.to(stacked.dtype)
        .view(1, bsz, 1, -1, 1)
    )
    denom = rna_mask.sum(dim=(2, 3)).clamp_min(1.0)  # [1, B, 1]
    attn = (stacked * rna_mask).sum(dim=(2, 3)) / denom  # [layer, B, L_protein]
    return attn.cpu().numpy()


def build_domain_mask(length: int, domains: list[dict]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for d in domains:
        s, e = int(d["start"]), int(d["end"])
        mask[max(0, s - 1) : min(length, e)] = True
    return mask


def attention_region_means(profile: np.ndarray, domain_mask: np.ndarray) -> tuple[float, float]:
    """Return (mean attention in domain, mean attention outside domain)."""
    dom = profile[domain_mask]
    nd = profile[~domain_mask]
    if dom.size == 0 or nd.size == 0:
        return float("nan"), float("nan")
    return float(dom.mean()), float(nd.mean())


def attention_ratio(profile: np.ndarray, domain_mask: np.ndarray) -> float:
    dom_mean, nd_mean = attention_region_means(profile, domain_mask)
    if not np.isfinite(dom_mean) or not np.isfinite(nd_mean):
        return float("nan")
    return float(dom_mean / (nd_mean + 1e-12))




def bootstrap_median_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(values))
    rng = np.random.default_rng(seed)
    boots = [float(np.median(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med, float(lo), float(hi)


def permutation_pvalue_ratio(
    profiles: np.ndarray,
    domain_mask: np.ndarray,
    observed_median: float,
    n_perm: int = 1000,
    seed: int = 0,
) -> float:
    """Permutation test: shuffle domain/non-domain labels on residues."""
    rng = np.random.default_rng(seed)
    n_dom = int(domain_mask.sum())
    n_total = domain_mask.size
    if n_dom == 0 or n_dom == n_total:
        return float("nan")
    null = []
    idx_all = np.arange(n_total)
    for _ in range(n_perm):
        perm_dom = np.zeros(n_total, dtype=bool)
        perm_dom[rng.choice(idx_all, size=n_dom, replace=False)] = True
        ratios = [attention_ratio(profiles[s].mean(axis=0), perm_dom) for s in range(profiles.shape[0])]
        null.append(float(np.median(ratios)))
    null = np.array(null)
    return float((null >= observed_median).mean())


def compute_significance(
    p_id: str,
    ratios: np.ndarray,
    profiles: np.ndarray,
    domain_mask: np.ndarray,
    seed: int,
) -> dict:
    ratios = ratios[np.isfinite(ratios)]
    med, ci_lo, ci_hi = bootstrap_median_ci(ratios, seed=seed)
    try:
        wilcox_p = float(stats.wilcoxon(ratios - 1.0, alternative="greater", zero_method="wilcox").pvalue)
    except Exception:
        wilcox_p = float("nan")
    perm_p = permutation_pvalue_ratio(profiles, domain_mask, med, seed=seed + 1)
    return {
        "p_id": p_id,
        "n_rna": int(len(ratios)),
        "median_ratio": med,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "wilcoxon_p_greater_than_1": wilcox_p,
        "perm_p_median": perm_p,
        "significant_wilcox_0.05": bool(wilcox_p < 0.05) if np.isfinite(wilcox_p) else False,
        "significant_perm_0.05": bool(perm_p < 0.05) if np.isfinite(perm_p) else False,
    }

def plot_fig8a(df: pd.DataFrame, out_png: Path) -> None:
    proteins = sorted(df["p_id"].unique())
    data = [df.loc[df["p_id"] == p, "ratio_layer_mean"].values for p in proteins]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.axhline(1.0, color="#888", ls="--", lw=1)
    ax.set_xticks(range(1, len(proteins) + 1))
    ax.set_xticklabels([p.replace("Human-", "") for p in proteins], rotation=15)
    ax.set_ylabel("Domain / non-domain attention ratio")
    ax.set_title("RBD cross-attention enrichment (pilot)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fig8b(layer_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for p_id, sub in layer_df.groupby("p_id"):
        ax.plot(sub["layer"], sub["ratio_mean"], marker="o", ms=4, label=p_id.replace("Human-", ""))
    ax.axhline(1.0, color="#888", ls="--", lw=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean domain / non-domain ratio")
    ax.set_title("Layer-wise RBD attention ratio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_profile(p_id: str, profile: np.ndarray, domain_mask: np.ndarray | None, out_png: Path) -> None:
    x = np.arange(1, len(profile) + 1)
    fig_w = max(14, len(profile) / 35)
    fig, ax = plt.subplots(figsize=(fig_w, 3))
    if domain_mask is not None and domain_mask.any():
        ax.fill_between(x, 0, profile.max(), where=domain_mask, color="#4daf4a", alpha=0.2, label="annotated domain")
        ax.legend(loc="upper right")
    ax.plot(x, profile, color="#2166ac", lw=0.8)
    ax.set_xlabel("Protein residue (1-based)")
    ax.set_ylabel("Mean cross-attention")
    ax.set_title(f"{p_id}: residue-level protein cross-attention")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_attention_tables(
    p_id: str,
    stacked: np.ndarray,
    out_dir: Path,
    domain_mask: np.ndarray | None,
) -> None:
    """Export residue-level attention for manual domain inspection."""
    # stacked: [N, layer, Lp]
    mean_over_samples = stacked.mean(axis=0)  # [layer, Lp]
    profile = mean_over_samples.mean(axis=0)
    std_over_samples = stacked.mean(axis=1).std(axis=0)

    prof_dir = out_dir / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)

    rows = {
        "residue": np.arange(1, profile.size + 1),
        "attention_mean": profile,
        "attention_std_across_rna": std_over_samples,
    }
    if domain_mask is not None:
        rows["in_annotated_domain"] = domain_mask.astype(int)
    pd.DataFrame(rows).to_csv(prof_dir / f"{p_id}_attention_mean.csv", index=False)

    layer_cols = {f"layer_{i}": mean_over_samples[i] for i in range(mean_over_samples.shape[0])}
    layer_cols["residue"] = np.arange(1, profile.size + 1)
    pd.DataFrame(layer_cols).to_csv(prof_dir / f"{p_id}_attention_by_layer.csv", index=False)


def list_test_proteins(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, usecols=["p_id"], low_memory=False)
    return sorted(df["p_id"].astype(str).str.strip().unique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(GEN_ROOT / "config/train.json"))
    ap.add_argument("--checkpoint", default=str(GEN_ROOT / "checkpoints/generator.pt"))
    ap.add_argument("--protein-h5", default=str(GEN_ROOT / "data/protein_embeddings.h5"))
    ap.add_argument("--rna-csv", default=str(GEN_ROOT / "data/test.csv"))
    ap.add_argument("--domain-json", default=str(PILOT_ROOT / "domain_annotations_test.json"))
    ap.add_argument("--proteins", nargs="+", default=None, help="Protein IDs; default: 4-protein demo list")
    ap.add_argument("--all-test-proteins", action="store_true", help="Use all p_id in --rna-csv")
    ap.add_argument("--skip-domain-ratio", action="store_true", help="Only export attention profiles (no ratio/significance)")
    ap.add_argument("--out-dir", default=str(PILOT_ROOT / "outputs/test"))
    ap.add_argument("--num-probes", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--t-value", type=float, default=0.0, help="0=clean RNA; >0 masked corruption")
    ap.add_argument("--min-rna-len", type=int, default=12, help="Filter probe RNAs shorter than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain_meta: dict = {}
    domain_json = Path(args.domain_json)
    if domain_json.exists():
        with open(domain_json, encoding="utf-8") as f:
            domain_meta = json.load(f)

    if args.all_test_proteins:
        proteins = list_test_proteins(Path(args.rna_csv))
    elif args.proteins:
        proteins = args.proteins
    else:
        proteins = ["Human-HNRNPK", "Human-U2AF1", "Human-PUM1", "Human-DKC1"]

    config = load_config(args.config)
    # Apply config-driven depth; sample.build_model_from_config uses utils.base_config default 12.
    from src import utils as gen_utils
    _n_layers = config.get("model", {}).get("num_hidden_layers")
    if _n_layers is not None:
        gen_utils.base_config["num_hidden_layers"] = int(_n_layers)
        print(f"[analyze] num_hidden_layers -> {int(_n_layers)} (GEN_ROOT={GEN_ROOT})", flush=True)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    use_bf16 = bool(config["train"].get("use_bf16", True)) and device.type == "cuda"

    tokenizer_dir = resolve_path(config["data"]["tokenizer_path"], GEN_ROOT, GEN_ROOT)
    tokenizer_runtime = build_tokenizer_runtime(
        Path(tokenizer_dir), device=device, token_length=1,
        allow_eos_in_content=bool(config["sample"].get("variable_length", True)),
    )
    tokenizer = tokenizer_runtime.tokenizer

    model = build_model_from_config(config, device, tokenizer_runtime)
    resume_model(args.checkpoint, model, device=device, strict=True)
    model.eval()

    protein_store = ProteinEmbeddingStore(Path(args.protein_h5))
    max_rna_nt = int(config["data"].get("max_generated_rna_bp", 101))

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    per_sample_rows: list[dict] = []
    layer_rows: list[dict] = []
    sig_rows: list[dict] = []
    index_rows: list[dict] = []
    protein_profiles: dict[str, np.ndarray] = {}

    for p_id in proteins:
        print(f"\n=== {p_id} ===")
        has_domains = (not args.skip_domain_ratio) and (p_id in domain_meta) and bool(domain_meta[p_id].get("domains"))
        domains = domain_meta[p_id]["domains"] if has_domains else []
        protein_emb = protein_store.get(p_id)
        lp, _ = protein_emb.shape
        domain_mask = build_domain_mask(lp, domains) if has_domains else None
        print(f"  protein_len={lp}, domain_residues={int(domain_mask.sum()) if domain_mask is not None else 0}, probes={args.num_probes}, has_domain={has_domains}")

        rnas = load_probe_rnas(Path(args.rna_csv), p_id, args.num_probes, args.seed, args.min_rna_len)
        print(f"  loaded {len(rnas)} RNAs (>={args.min_rna_len}bp)")
        probe_ds = ProbeDataset(rnas, p_id)
        protein_dim = protein_emb.shape[1]

        def collate_probe(batch):
            lp_ph = protein_emb.shape[0]
            for sample in batch:
                sample["protein_emb"] = torch.zeros(lp_ph, protein_dim)
                sample["protein_len"] = lp_ph
            return rna_rbp_collate_fn(batch, tokenizer, append_eos=False, max_rna_nt=max_rna_nt)

        loader = DataLoader(probe_ds, batch_size=min(args.batch_size, len(rnas)), shuffle=False, collate_fn=collate_probe)

        layer_accum: list[np.ndarray] = []

        for batch_idx, batch in enumerate(loader):
            x0 = batch["rna_input_ids_clean"].to(device)
            rna_attention_mask = batch["rna_attention_mask"].to(device)
            b = x0.size(0)
            protein_cond = protein_emb.unsqueeze(0).expand(b, -1, -1).to(device)
            protein_attention_mask = torch.ones(b, protein_emb.shape[0], dtype=torch.long, device=device)

            if args.t_value <= 0:
                xt = x0
            else:
                t = torch.full((b,), args.t_value, device=device)
                xt, _, _ = build_xt_and_labels(
                    x0=x0, rna_attention_mask=rna_attention_mask,
                    mask_token_id=tokenizer.mask_token_id, t=t,
                    special_token_ids=[tokenizer.pad_token_id, tokenizer.cls_token_id],
                    force_at_least_one_mask=True,
                )

            with autocast_ctx:
                attn_layers = collect_cross_attention(
                    model, input_ids=xt, rna_attention_mask=rna_attention_mask,
                    protein_cond=protein_cond, protein_attention_mask=protein_attention_mask,
                )

            for i in range(b):
                sample_profiles = attn_layers[:, i, :]  # [layer, Lp]
                if domain_mask is not None:
                    ratio_layers = [attention_ratio(sample_profiles[l], domain_mask) for l in range(sample_profiles.shape[0])]
                    # Layer-averaged profile → absolute RBD / non-RBD means (matches perm-test profile)
                    layer_mean_profile = sample_profiles.mean(axis=0)
                    mean_rbd, mean_non = attention_region_means(layer_mean_profile, domain_mask)
                    per_sample_rows.append({
                        "p_id": p_id, "batch_idx": batch_idx, "sample_idx": batch_idx * args.batch_size + i,
                        "ratio_layer_mean": float(np.nanmean(ratio_layers)),
                        "ratio_layer_median": float(np.nanmedian(ratio_layers)),
                        "mean_rbd": mean_rbd,
                        "mean_non": mean_non,
                        "t_value": args.t_value,
                    })
                layer_accum.append(sample_profiles)

        stacked = np.stack(layer_accum, axis=0)  # [N, layer, Lp]
        mean_profile = stacked.mean(axis=(0, 1))
        plot_profile(p_id, mean_profile, domain_mask, out_dir / "profiles" / f"{p_id}_attention_profile.png")
        save_attention_tables(p_id, stacked, out_dir, domain_mask)

        protein_profiles[p_id] = stacked
        index_rows.append({
            "p_id": p_id,
            "protein_len": lp,
            "n_rna": stacked.shape[0],
            "has_domain_annotation": has_domains,
            "profile_png": str(out_dir / "profiles" / f"{p_id}_attention_profile.png"),
            "profile_csv": str(out_dir / "profiles" / f"{p_id}_attention_mean.csv"),
        })

        if domain_mask is not None:
            for layer_idx in range(stacked.shape[1]):
                ratios = [attention_ratio(stacked[s, layer_idx], domain_mask) for s in range(stacked.shape[0])]
                layer_rows.append({
                    "p_id": p_id, "layer": layer_idx,
                    "ratio_mean": float(np.nanmean(ratios)),
                    "ratio_median": float(np.nanmedian(ratios)),
                    "ratio_std": float(np.nanstd(ratios)),
                })

    protein_store.close()

    pd.DataFrame(index_rows).to_csv(out_dir / "protein_index.csv", index=False)

    if per_sample_rows:
        sample_df = pd.DataFrame(per_sample_rows)
        layer_df = pd.DataFrame(layer_rows)
        for p_id in sample_df["p_id"].unique():
            if p_id not in domain_meta:
                continue
            ratios = sample_df.loc[sample_df["p_id"] == p_id, "ratio_layer_mean"].values
            sig_rows.append(compute_significance(p_id, ratios, protein_profiles[p_id], build_domain_mask(
                protein_profiles[p_id].shape[-1], domain_meta[p_id]["domains"]
            ), args.seed))
        sig_df = pd.DataFrame(sig_rows)
        sample_df.to_csv(out_dir / "per_sample_ratios.csv", index=False)
        layer_df.to_csv(out_dir / "per_layer_ratios.csv", index=False)
        sig_df.to_csv(out_dir / "significance_summary.csv", index=False)
        if len(sample_df["p_id"].unique()) >= 2:
            plot_fig8a(sample_df, out_dir / "fig8a_ratio_violin.png")
            plot_fig8b(layer_df, out_dir / "fig8b_ratio_by_layer.png")
        print(f"\nWrote ratio tables for {len(sig_rows)} annotated protein(s)")

    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config, "checkpoint": args.checkpoint,
            "rna_csv": args.rna_csv, "num_probes": args.num_probes,
            "t_value": args.t_value, "proteins": proteins,
            "skip_domain_ratio": args.skip_domain_ratio,
        }, f, indent=2)

    print(f"\nDone: {len(proteins)} proteins -> {out_dir / 'profiles'}")
    print(f"Index: {out_dir / 'protein_index.csv'}")


if __name__ == "__main__":
    main()
