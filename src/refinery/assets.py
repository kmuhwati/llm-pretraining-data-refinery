"""Dagster asset graph for the refinery.

    extracted_docs -> exact_deduped -> quality_filtered -> minhash_signatures -> near_deduped
                   -> decontaminated_redacted ---------------------------------------------\
    extracted_docs -> quality_classifier --------------------------------------------------- > scored_corpus -> token_shards

Each asset writes its outputs to data/ and records counts and timings as
Dagster metadata (also saved to data/stages/_stats/ for the report). Run the
whole graph with `python scripts/run_pipeline.py`, or browse it with
`dagster dev -m refinery.assets`.
"""
from __future__ import annotations

import json
import random
import time

import numpy as np
import pandas as pd
from dagster import Definitions, MaterializeResult, MetadataValue, asset

from . import config as C
from . import stages
from .classifier import QualityClassifier, fetch_wikipedia_leads
from .tokenize_shards import write_shards

STATS = C.STAGES / "_stats"


def _record(name: str, stats: dict, seconds: float) -> MaterializeResult:
    STATS.mkdir(parents=True, exist_ok=True)
    stats = {**stats, "seconds": round(seconds, 1)}
    (STATS / f"{name}.json").write_text(json.dumps(stats, indent=2, default=float), encoding="utf-8")
    meta = {k: (MetadataValue.int(int(v)) if isinstance(v, (int, np.integer)) else MetadataValue.float(float(v)))
            for k, v in stats.items() if isinstance(v, (int, float, np.integer, np.floating))}
    return MaterializeResult(metadata=meta)


@asset(group_name="refinery", description="WET text extraction, URL blocklist and fastText English ID")
def extracted_docs() -> MaterializeResult:
    t0 = time.time()
    per_file = stages.extract(stages.wet_files())
    per_file.to_csv(C.TABLES / "extract_per_file.csv", index=False)
    return _record("extracted_docs", {"files": len(per_file), "docs_in": int(per_file.docs_in.sum()),
                                      "docs_out": int(per_file.docs_out.sum()), "raw_gb": per_file.raw_chars.sum() / 1e9,
                                      "cpu_seconds": float(per_file.seconds.sum())}, time.time() - t0)


@asset(deps=[extracted_docs], group_name="refinery", description="Exact duplicates on normalised text (global)")
def exact_deduped() -> MaterializeResult:
    t0 = time.time()
    return _record("exact_deduped", stages.exact_dedup(stages.wet_files()), time.time() - t0)


@asset(deps=[exact_deduped], group_name="refinery", description="C4 cleaning, Gopher repetition/quality, FineWeb filters")
def quality_filtered() -> MaterializeResult:
    t0 = time.time()
    per_file = stages.quality(stages.wet_files())
    return _record("quality_filtered", {"docs_in": int(per_file.docs_in.sum()), "docs_out": int(per_file.docs_out.sum())}, time.time() - t0)


@asset(deps=[quality_filtered], group_name="refinery", description="112-permutation MinHash signatures on word 5-grams")
def minhash_signatures() -> MaterializeResult:
    t0 = time.time()
    return _record("minhash_signatures", {"docs": stages.signatures(stages.wet_files())}, time.time() - t0)


@asset(deps=[minhash_signatures], group_name="refinery", description="LSH (14x8) near-duplicate clustering (global)")
def near_deduped() -> MaterializeResult:
    t0 = time.time()
    return _record("near_deduped", stages.near_dedup(stages.wet_files()), time.time() - t0)


@asset(deps=[near_deduped], group_name="refinery", description="13-gram benchmark decontamination and PII redaction")
def decontaminated_redacted() -> MaterializeResult:
    t0 = time.time()
    per_file = stages.decontaminate_and_redact(stages.wet_files())
    totals = {c: int(per_file[c].sum()) for c in per_file.columns if c.startswith("pii_")}
    return _record("decontaminated_redacted", {"docs_in": int(per_file.docs_in.sum()), "docs_out": int(per_file.docs_out.sum()), **totals},
                   time.time() - t0)


@asset(deps=[extracted_docs], group_name="refinery", description="fastText quality classifier: Wikipedia leads vs random English web")
def quality_classifier() -> MaterializeResult:
    t0 = time.time()
    positives = fetch_wikipedia_leads()
    files = stages.wet_files()
    per_file = -(-len(positives) // len(files))  # an equal random share from every file, read one file at a time
    negatives = []
    for i, path in enumerate(files):
        texts = stages.read_stage("extract", [path], columns=["text"]).text.tolist()
        negatives += random.Random(C.SEED + i).sample(texts, per_file)
    negatives = negatives[:len(positives)]
    clf, metrics = QualityClassifier.train(positives, negatives)
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    clf.save(C.OUTPUT / "quality_classifier.bin")
    return _record("quality_classifier", metrics, time.time() - t0)


@asset(deps=[decontaminated_redacted, quality_classifier], group_name="refinery",
       description="Score every document and write the corpus as Parquet")
def scored_corpus() -> MaterializeResult:
    import fasttext

    t0 = time.time()
    from .classifier import QualityClassifier as QC

    clf = QC(fasttext.load_model(str(C.OUTPUT / "quality_classifier.bin")))
    df = stages.read_stage("clean", stages.wet_files())
    df["quality_score"] = [clf.score(t) for t in df.text]
    cutoff = float(np.quantile(df.quality_score, 1 - C.HQ_TOP_FRACTION))
    df["high_quality"] = df.quality_score >= cutoff
    df["n_words"] = df.text.str.split().str.len()
    out = C.OUTPUT / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.parquet"):
        old.unlink()
    for i, part in enumerate(np.array_split(df.index, max(1, len(df) // 20_000))):
        df.loc[part].to_parquet(out / f"part-{i:05d}.parquet", index=False)
    return _record("scored_corpus", {"docs": len(df), "hq_docs": int(df.high_quality.sum()), "hq_cutoff": cutoff,
                                     "words": int(df.n_words.sum())}, time.time() - t0)


@asset(deps=[scored_corpus], group_name="refinery", description="tiktoken o200k_base shards (all + high-quality subset)")
def token_shards() -> MaterializeResult:
    t0 = time.time()
    df = pd.read_parquet(C.OUTPUT / "corpus", columns=["text", "high_quality"])
    all_m = write_shards(df.text.tolist(), C.OUTPUT / "tokens", "all")
    hq_m = write_shards(df.loc[df.high_quality, "text"].tolist(), C.OUTPUT / "tokens", "hq")
    return _record("token_shards", {"tokens_all": all_m["tokens"], "tokens_hq": hq_m["tokens"], "shards_all": len(all_m["shards"]),
                                    "shards_hq": len(hq_m["shards"]), "mean_tokens_per_doc": all_m["mean_tokens_per_doc"]},
                   time.time() - t0)


ALL_ASSETS = [extracted_docs, exact_deduped, quality_filtered, minhash_signatures, near_deduped,
              decontaminated_redacted, quality_classifier, scored_corpus, token_shards]
defs = Definitions(assets=ALL_ASSETS)
