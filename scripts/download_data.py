"""Download everything the refinery needs (about 1.1 GB with the default 16 WET files).

- Common Crawl WET files (extracted plain text) from one crawl, spread evenly
  across its 100,000 files so the sample covers the whole crawl window.
- Public benchmark test sets used for decontamination.
- fastText's 176-language identification model.

    python scripts/download_data.py --files 16
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import io
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CRAWL = "CC-MAIN-2026-39"
CC = "https://data.commoncrawl.org/"
UA = {"User-Agent": "kmuhwati-llm-data-refinery/1.0 (https://github.com/kmuhwati)"}

BENCHMARKS = {
    # name: (url, local file)
    "mmlu": ("https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet", "mmlu_test.parquet"),
    "arc_challenge": ("https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet", "arc_challenge_test.parquet"),
    "gsm8k": ("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl", "gsm8k_test.jsonl"),
    "truthfulqa": ("https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv", "truthfulqa.csv"),
    "humaneval": ("https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz", "humaneval.jsonl.gz"),
}
FASTTEXT_LID = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def fetch(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size


def wet_paths(n: int) -> list[str]:
    listing = RAW / "wet.paths"
    if not listing.exists():
        data = urllib.request.urlopen(urllib.request.Request(f"{CC}crawl-data/{CRAWL}/wet.paths.gz", headers=UA), timeout=60).read()
        listing.write_text(gzip.decompress(data).decode())
    paths = listing.read_text().split()
    step = len(paths) // n
    return [paths[i * step] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=16)
    args = ap.parse_args()

    fetch(FASTTEXT_LID, RAW / "lid.176.ftz")
    for name, (url, local) in BENCHMARKS.items():
        print(f"{name}: {fetch(url, RAW / 'benchmarks' / local):,} bytes")

    jobs = {p: RAW / "wet" / Path(p).name for p in wet_paths(args.files)}
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, CC + p, dest): p for p, dest in jobs.items()}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            print(f"[{i}/{len(jobs)}] {Path(futures[fut]).name}: {fut.result() / 1e6:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
