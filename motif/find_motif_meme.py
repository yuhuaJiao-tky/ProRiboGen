#!/usr/bin/env python3
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_meme(fasta_file, output_dir, minw=6, maxw=8, nmotifs=4):
    """Run MEME on one FASTA file."""
    background_fasta_file = "background.txt"
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "meme", fasta_file,
        "-oc", output_dir,
        "-rna",
        "-minw", str(minw),
        "-maxw", str(maxw),
        "-nmotifs", str(nmotifs),
        "-markov_order", "0",
        "-mod", "zoops",
        "-objfun", "classic",
        "-nostatus",
        "-bfile", background_fasta_file,
    ]
    try:
        subprocess.run(command, check=True)
        print(f"OK: {fasta_file}")
    except subprocess.CalledProcessError as e:
        print(f"MEME failed on {fasta_file}: {e}")


def batch_process_fasta(input_folder, output_folder, max_threads=4):
    """Run MEME on every FASTA in input_folder."""
    fasta_files = [f for f in os.listdir(input_folder) if f.endswith(".fasta")]
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for filename in fasta_files:
            fasta_file = os.path.join(input_folder, filename)
            output_subfolder = os.path.join(output_folder, filename.split(".")[0])
            futures.append(executor.submit(run_meme, fasta_file, output_subfolder))
        for future in as_completed(futures):
            future.result()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch MEME on FASTA files")
    parser.add_argument("input_folder", type=str, help="Folder of FASTA files")
    parser.add_argument("output_folder", type=str, help="Output root")
    parser.add_argument("--threads", type=int, default=60, help="Max worker threads")
    args = parser.parse_args()
    batch_process_fasta(args.input_folder, args.output_folder, args.threads)


if __name__ == "__main__":
    main()
