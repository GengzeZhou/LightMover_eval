#!/usr/bin/env python3
"""
Download the ObjectMover benchmark dataset from Hugging Face into ./data.

Usage:
    # use HF_TOKEN env var, or pass --token
    python download_data.py
    python download_data.py --token hf_xxx
    python download_data.py --subset ObjMove-A
    python download_data.py --subset Lightmove-A
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Andyx/ObjectMover-Benchmark"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID,
                        help="Hugging Face dataset repo ID")
    parser.add_argument("--subset", default=None,
                        help="Optional subset to download (e.g. ObjMove-A, "
                             "ObjMove-B, Lightmove-A). Default: full repo.")
    parser.add_argument("--output", default="./data",
                        help="Output directory (default: ./data)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HF access token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = None
    if args.subset:
        allow_patterns = [f"{args.subset}/**"]

    print(f"Downloading {args.repo_id} -> {output_dir}")
    if args.subset:
        print(f"  subset: {args.subset}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        token=args.token,
        allow_patterns=allow_patterns,
    )
    print("Done.")


if __name__ == "__main__":
    main()
