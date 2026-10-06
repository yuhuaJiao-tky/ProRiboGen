#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_MOTIF_DIR = Path(__file__).resolve().parent
_DEFAULT_BFILE = _MOTIF_DIR / "background.txt"


def run_meme(
    fasta_file: Path,
    output_dir: Path,
    bfile: Path,
    minw: int = 6,
    maxw: int = 8,
    nmotifs: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
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
    subprocess.run(command, check=True)
    meme_txt = output_dir / "meme.txt"
    if not meme_txt.is_file():
        raise FileNotFoundError(f"MEME finished but no meme.txt in {output_dir}")
    print(f"OK: {fasta_file.name} -> {meme_txt}")


def batch_process_fasta(
    input_folder: Path,
    output_folder: Path,
    max_threads: int,
    bfile: Path,
) -> None:
    fasta_files = sorted(input_folder.glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"no *.fasta in {input_folder}")
    ok = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(
                run_meme,
                fa,
                output_folder / fa.stem,
                bfile,
            ): fa
            for fa in fasta_files
        }
        for fut in as_completed(futures):
            fa = futures[fut]
            try:
                fut.result()
                ok += 1
            except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
                failed.append(f"{fa.name}: {e}")
            except Exception as e:
                failed.append(f"{fa.name}: {e}")
    print(f"MEME done: {ok}/{len(fasta_files)} wrote meme.txt under {output_folder}")
    if failed:
        print("Failures:", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
    if ok == 0:
        raise SystemExit(
            "No meme.txt produced. Check that `meme` is on PATH "
            "(separate MEME env, not the PyTorch env) and that background.txt exists."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch MEME on FASTA files")
    parser.add_argument("input_folder", type=Path, help="Folder of FASTA files")
    parser.add_argument("output_folder", type=Path, help="Output root (one subdir per FASTA)")
    parser.add_argument("--threads", type=int, default=8, help="Max worker threads")
    parser.add_argument(
        "--bfile",
        type=Path,
        default=_DEFAULT_BFILE,
        help="MEME Markov background (default: motif/background.txt next to this script)",
    )
    args = parser.parse_args()
    if shutil.which("meme") is None:
        raise SystemExit(
            "`meme` not found on PATH. Activate your MEME environment "
            "(e.g. conda activate meme_new) and retry. Do not run this in the PyTorch-only env."
        )
    bfile = args.bfile.resolve()
    if not bfile.is_file():
        raise SystemExit(f"background file missing: {bfile}")
    inp = args.input_folder.expanduser().resolve()
    out = args.output_folder.expanduser().resolve()
    if not inp.is_dir():
        raise SystemExit(f"FASTA folder missing: {inp}")
    batch_process_fasta(inp, out, args.threads, bfile)


if __name__ == "__main__":
    main()
