#!/usr/bin/env python3
"""Turn generated FASTA (one file per protein) into a probe CSV for analyze_rbd_cross_attention.py.

Expected: <fasta_dir>/Human-PUM1.fasta with RNA sequences. Output columns: p_id,rna
"""
from __future__ import annotations

import argparse
from pathlib import Path


def iter_fasta(path: Path):
    header, parts = None, []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and parts:
                    yield header, "".join(parts)
                header = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.upper().replace("T", "U"))
        if header is not None and parts:
            yield header, "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--p-id-from", choices=["filename", "header"], default="filename")
    args = ap.parse_args()
    rows = []
    for fa in sorted(args.fasta_dir.glob("*.fasta")) + sorted(args.fasta_dir.glob("*.fa")):
        pid_file = fa.stem
        for header, seq in iter_fasta(fa):
            seq = "".join(ch for ch in seq if ch in "ACGU")
            if not seq:
                continue
            pid = pid_file if args.p_id_from == "filename" else header
            rows.append(f"{pid},{seq}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("p_id,rna\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
