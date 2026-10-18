"""Benchmark decontamination by 13-gram overlap (the GPT-3 / Brown et al., 2020 recipe).

Every 13-word window of each benchmark item (question plus reference answer)
is hashed into a set. A web document sharing any 13-gram with a benchmark is
removed, so a model trained on the corpus cannot have read those test items.
Thirteen words is long enough that an accidental match is very unlikely.
"""
from __future__ import annotations

import gzip
import json
import re

import numpy as np
import pandas as pd
import xxhash

from . import config as C

_WORD = re.compile(r"[a-z0-9]+")


def ngram_hashes(text: str, n: int = C.DECONTAM_NGRAM) -> set[int]:
    words = _WORD.findall(text.lower())
    return {xxhash.xxh64_intdigest(" ".join(words[i:i + n]).encode()) for i in range(len(words) - n + 1)}


def benchmark_texts() -> dict[str, list[str]]:
    d = C.BENCH_DIR
    out: dict[str, list[str]] = {}
    mmlu = pd.read_parquet(d / "mmlu_test.parquet")
    out["MMLU"] = [f"{q} {' '.join(map(str, ch))}" for q, ch in zip(mmlu.question, mmlu.choices)]
    arc = pd.read_parquet(d / "arc_challenge_test.parquet")
    out["ARC-Challenge"] = [f"{q} {' '.join(ch['text'])}" for q, ch in zip(arc.question, arc.choices)]
    out["GSM8K"] = [f"{r['question']} {r['answer']}" for r in map(json.loads, (d / "gsm8k_test.jsonl").read_text(encoding="utf-8").splitlines())]
    tqa = pd.read_csv(d / "truthfulqa.csv")
    out["TruthfulQA"] = [f"{q} {a}" for q, a in zip(tqa["Question"], tqa["Best Answer"])]
    with gzip.open(d / "humaneval.jsonl.gz", "rt", encoding="utf-8") as f:
        out["HumanEval"] = [json.loads(line)["prompt"] for line in f]
    return out


class Decontaminator:
    def __init__(self, benchmarks: dict[str, list[str]] | None = None):
        benchmarks = benchmarks if benchmarks is not None else benchmark_texts()
        self.index: dict[int, str] = {}
        self.items = {name: len(texts) for name, texts in benchmarks.items()}
        for name, texts in benchmarks.items():
            for text in texts:
                for h in ngram_hashes(text):
                    self.index.setdefault(h, name)

    def match(self, text: str) -> str | None:
        """Name of the first benchmark sharing a 13-gram with `text`, else None."""
        for h in ngram_hashes(text):
            hit = self.index.get(h)
            if hit:
                return hit
        return None

    def __len__(self) -> int:
        return len(self.index)
