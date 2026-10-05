#!/usr/bin/env python3
"""
Convert MEME .txt files in a flat folder to HOMER .motif matrices.
Write to a separate directory. Does not modify create_motif_logo.ipynb.

Each foo.txt -> output_dir/foo.motif; RBP label prefix is the stem.
Keep motifs with E ≤ 0.05 (same as the notebook).
"""
import argparse
import os
import re
from typing import Dict, List, Optional


def create_homer_header(orig_name: str, new_label: str) -> str:
    return f">{orig_name}\t{new_label}\t+"


def extract_motifs_to_homer(
    meme_file: str, output_dir: str, rbp_name: Optional[str] = None
) -> None:
    if rbp_name is None:
        rbp_name = os.path.basename(os.path.dirname(meme_file))
    rbp_label_prefix = rbp_name
    motif_count = 0

    with open(meme_file, "r") as file:
        lines = file.readlines()

    motif_name = None
    matrix_lines: List[str] = []
    capturing = False
    significant_motifs: List[Dict] = []

    for line in lines:
        motif_match = re.search(r"^MOTIF (\S+)", line)
        if motif_match:
            if motif_name and matrix_lines:
                try:
                    e_value = float(matrix_lines[0].split("E=")[1].strip())
                    if e_value <= 0.05:
                        motif_count += 1
                        significant_motifs.append(
                            {
                                "orig_name": motif_name,
                                "label": f"{rbp_label_prefix}_{motif_count}",
                                "matrix": matrix_lines[1:],
                            }
                        )
                except Exception:
                    pass
            matrix_lines = []
            capturing = False
            motif_name = motif_match.group(1)

        matrix_header = re.search(
            r"letter-probability matrix: .* E= ([0-9eE.+-]+)", line
        )
        if matrix_header:
            e_value = float(matrix_header.group(1))
            capturing = e_value <= 0.05
            if capturing:
                matrix_lines.append(line.strip())
            continue

        if capturing:
            line_clean = line.strip()
            if re.match(r"^([0-9eE.+-]+\s+){3}[0-9eE.+-]+$", line_clean):
                matrix_lines.append(line_clean)
            else:
                capturing = False

    if motif_name and matrix_lines:
        try:
            e_value = float(matrix_lines[0].split("E=")[1].strip())
            if e_value <= 0.05:
                motif_count += 1
                significant_motifs.append(
                    {
                        "orig_name": motif_name,
                        "label": f"{rbp_label_prefix}_{motif_count}",
                        "matrix": matrix_lines[1:],
                    }
                )
        except Exception:
            pass

    if significant_motifs:
        save_homer_motifs(rbp_name, significant_motifs, output_dir)


def save_homer_motifs(rbp_name: str, motifs: List[Dict], output_dir: str) -> None:
    output_content: List[str] = []
    for motif in motifs:
        output_content.append(create_homer_header(motif["orig_name"], motif["label"]))
        for row in motif["matrix"]:
            row = re.sub(r"\s+", "\t", row.strip())
            output_content.append(row)

    output_file = os.path.join(output_dir, f"{rbp_name}.motif")
    with open(output_file, "w") as out:
        out.write("\n".join(output_content) + "\n")

    print(f"saved {len(motifs)} motifs for {rbp_name} -> {output_file}")


def batch_process_txt_folder_to_homer(txt_folder: str, output_dir: str) -> None:
    if not os.path.isdir(txt_folder):
        raise NotADirectoryError(txt_folder)
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(txt_folder)):
        path = os.path.join(txt_folder, filename)
        if not os.path.isfile(path):
            continue
        if not filename.lower().endswith(".txt"):
            continue
        stem = os.path.splitext(filename)[0]
        print(f"processing: {path}")
        extract_motifs_to_homer(path, output_dir, rbp_name=stem)


def main() -> None:
    p = argparse.ArgumentParser(
        description="MEME .txt in a flat folder -> HOMER .motif"
    )
    p.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="folder of MEME .txt files",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="output directory for .motif files",
    )
    args = p.parse_args()
    batch_process_txt_folder_to_homer(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
