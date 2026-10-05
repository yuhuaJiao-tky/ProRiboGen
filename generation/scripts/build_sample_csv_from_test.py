#!/usr/bin/env python3
"""Build GPU-sharded sample CSVs from a test pair table (one row per unique p_id)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test-csv",
        default="data/test.csv",
        help="Test CSV with a p_id column",
    )
    ap.add_argument("--out-dir", default="data/csvs", help="Output directory")
    ap.add_argument("--num-sequences", type=int, default=2048)
    ap.add_argument("--length-bp", type=int, default=101)
    ap.add_argument(
        "--omit-length-bp",
        action="store_true",
        help="Do not write length_bp; sample.py infers length from train r_len",
    )
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--prefix", default="sample_test", help="Filename prefix")
    args = ap.parse_args()

    test_path = Path(args.test_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_path)
    if "p_id" not in df.columns:
        raise SystemExit(f"Missing p_id column: {test_path}")

    p_ids = sorted(df["p_id"].astype(str).unique())
    rows = []
    for p_id in p_ids:
        row = {"p_id": p_id, "num_sequences": args.num_sequences}
        if not args.omit_length_bp:
            row["length_bp"] = args.length_bp
        rows.append(row)
    master = pd.DataFrame(rows)
    master_path = out_dir / f"{args.prefix}.csv"
    master.to_csv(master_path, index=False)

    n_gpus = max(1, args.num_gpus)
    for i in range(n_gpus):
        shard = rows[i::n_gpus]
        shard_path = out_dir / f"{args.prefix}_gpu{i}.csv"
        pd.DataFrame(shard).to_csv(shard_path, index=False)

    print(f"Wrote {len(p_ids)} RBPs -> {master_path}")
    for i in range(n_gpus):
        n = len(rows[i::n_gpus])
        print(f"  gpu{i}: {n} RBPs -> {out_dir / f'{args.prefix}_gpu{i}.csv'}")


if __name__ == "__main__":
    main()
