#!/usr/bin/env python3
"""把 find_motif_meme 产出的 motifs/<p_id>/meme.txt 转为 HOMER .motif。"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from meme_txt_folder_to_homer import batch_process_txt_folder_to_homer


def convert_meme_tree(motifs_root: Path, homer_out: Path) -> int:
    motifs_root = motifs_root.resolve()
    homer_out = homer_out.resolve()
    homer_out.mkdir(parents=True, exist_ok=True)
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        flat = Path(tmp)
        for meme in motifs_root.rglob("meme.txt"):
            p_id = meme.parent.name
            shutil.copy(meme, flat / f"{p_id}.txt")
            n += 1
        if n == 0:
            raise FileNotFoundError(f"未找到 meme.txt: {motifs_root}")
        batch_process_txt_folder_to_homer(str(flat), str(homer_out))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("motifs_root", type=Path)
    ap.add_argument("homer_out", type=Path)
    args = ap.parse_args()
    n = convert_meme_tree(args.motifs_root, args.homer_out)
    print(f"converted {n} proteins -> {args.homer_out}")


if __name__ == "__main__":
    main()
