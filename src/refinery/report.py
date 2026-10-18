"""Figures, tables, RESULTS.md, metrics.json and the dataset card.

Only URLs and statistics are published. No page text from the crawl is copied
into the repository, so scraped personal content never ends up in git.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import plotstyle, stages

STAGE_ORDER = ["extract", "exact", "quality", "near", "clean"]


def _stats(name: str) -> dict:
    return json.loads((C.STAGES / "_stats" / f"{name}.json").read_text(encoding="utf-8"))


def build_report() -> dict:
    plotstyle.apply()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    s = {name: _stats(name) for name in ["extracted_docs", "exact_deduped", "quality_filtered", "minhash_signatures",
                                         "near_deduped", "decontaminated_redacted", "quality_classifier", "scored_corpus", "token_shards"]}
    drops = pd.concat([stages.read_drops(st) for st in STAGE_ORDER], ignore_index=True)
    drops["family"] = drops.reason.str.split(":").str[0]
    drops.groupby(["stage", "reason"]).size().rename("documents").reset_index() \
        .sort_values("documents", ascending=False).to_csv(C.TABLES / "drop_reasons.csv", index=False)

    raw = s["extracted_docs"]["docs_in"]
    url_drops = int((drops.family == "url").sum())
    lang_drops = int((drops.family == "lang").sum())
    funnel = pd.DataFrame([
        ("Raw web pages (16 WET files)", raw),
        ("URL blocklist", raw - url_drops),
        ("English (fastText ≥ 0.65)", raw - url_drops - lang_drops),
        ("Exact dedup", s["exact_deduped"]["docs_out"]),
        ("Quality filters (C4, Gopher, FineWeb)", s["quality_filtered"]["docs_out"]),
        ("Near dedup (MinHash LSH)", s["near_deduped"]["docs_out"]),
        ("Benchmark decontamination", s["decontaminated_redacted"]["docs_out"]),
    ], columns=["step", "documents"])
    funnel["share_of_raw"] = funnel.documents / raw
    funnel.to_csv(C.TABLES / "funnel.csv", index=False)

    corpus = pd.read_parquet(C.OUTPUT / "corpus", columns=["doc_id", "url", "domain", "quality_score", "high_quality",
                                                           "cluster_size", "pii_redactions", "n_words", "cc_language"])
    # ---------------- figures
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(funnel))[::-1]
    ax.barh(y, funnel.documents, color=[plotstyle.NEUTRAL] * 3 + [plotstyle.SERIES[0]] * 3 + [plotstyle.SERIES[2]], height=0.6)
    for yi, (n, share) in zip(y, zip(funnel.documents, funnel.share_of_raw)):
        ax.annotate(f"{n:,}  ({100 * share:.1f}%)", (n, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(y, funnel.step)
    ax.set_xlim(0, raw * 1.25)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Documents remaining")
    ax.set_title(f"From {raw:,} raw web pages to {funnel.documents.iloc[-1]:,} training documents")
    plotstyle.save(fig, C.FIGURES / "01_funnel.png")

    q = drops[drops.stage == "quality"].reason.value_counts().head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(q.index, q.values, color=plotstyle.SERIES[1], height=0.6)
    ax.set_xlabel("Documents rejected")
    ax.set_title("Why English pages failed the quality filters")
    plotstyle.save(fig, C.FIGURES / "02_quality_reasons.png")

    langs = drops[drops.family == "lang"].reason.str.split(":").str[1].value_counts().head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(langs.index, langs.values, color=plotstyle.NEUTRAL, height=0.6)
    ax.set_xlabel("Documents")
    ax.set_title("Most common non-English languages removed")
    plotstyle.save(fig, C.FIGURES / "03_languages_removed.png")

    sizes = pd.Series(corpus.cluster_size).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(sizes.index, sizes.values, color=plotstyle.SERIES[0], s=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Near-duplicate cluster size (documents)")
    ax.set_ylabel("Clusters")
    ax.set_title("Most pages are unique; a long tail repeats many times")
    plotstyle.save(fig, C.FIGURES / "04_cluster_sizes.png")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(np.log10(corpus.quality_score.clip(lower=1e-5)), bins=60, color=plotstyle.SERIES[0])
    cut = np.log10(s["scored_corpus"]["hq_cutoff"])
    ax.axvline(cut, color=plotstyle.INK_2, ls=":", lw=1)
    ax.annotate("top 30% →\nhigh-quality subset", (cut, ax.get_ylim()[1] * 0.85), xytext=(5, 0),
                textcoords="offset points", fontsize=9, color=plotstyle.INK_2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{10 ** v:g}"))
    ax.set_xlabel("Quality score, log scale (P[reference-quality prose])")
    ax.set_ylabel("Documents")
    ax.set_title("Model-based quality scores of the refined corpus")
    plotstyle.save(fig, C.FIGURES / "05_quality_scores.png")

    doms = corpus.domain.str.replace(r"^www\.", "", regex=True).value_counts().head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(doms.index, doms.values, color=plotstyle.SERIES[2], height=0.6)
    ax.set_xlabel("Documents in refined corpus")
    ax.set_title("Top source domains after refining")
    plotstyle.save(fig, C.FIGURES / "06_top_domains.png")

    timing = pd.DataFrame([
        ("Extract + URL + language", s["extracted_docs"]["seconds"], raw),
        ("Exact dedup", s["exact_deduped"]["seconds"], s["exact_deduped"]["docs_in"]),
        ("Quality filters", s["quality_filtered"]["seconds"], s["quality_filtered"]["docs_in"]),
        ("MinHash signatures", s["minhash_signatures"]["seconds"], s["minhash_signatures"]["docs"]),
        ("LSH clustering", s["near_deduped"]["seconds"], s["near_deduped"]["docs_in"]),
        ("Decontamination + PII", s["decontaminated_redacted"]["seconds"], s["decontaminated_redacted"]["docs_in"]),
        ("Tokenise + shard", s["token_shards"]["seconds"], s["scored_corpus"]["docs"]),
    ], columns=["stage", "seconds", "documents"])
    timing["docs_per_second"] = timing.documents / timing.seconds.clip(lower=0.1)
    timing.to_csv(C.TABLES / "stage_timing.csv", index=False)

    # ---------------- tables a reviewer can audit (URLs only)
    examples = corpus.sort_values("quality_score")
    pd.concat([examples.head(15).assign(group="lowest score"), examples.tail(15).assign(group="highest score")])[
        ["group", "quality_score", "url"]].to_csv(C.TABLES / "quality_score_examples.csv", index=False)
    contam = drops[drops.stage == "clean"].reason.str.split(":").str[1].value_counts()
    contam.rename_axis("benchmark").rename("documents_removed").to_csv(C.TABLES / "contamination_by_benchmark.csv")

    metrics = {
        "crawl": "CC-MAIN-2026-39", "wet_files": s["extracted_docs"]["files"], "raw_documents": raw,
        "raw_text_gb": s["extracted_docs"]["raw_gb"], "final_documents": int(funnel.documents.iloc[-1]),
        "final_share_of_raw": float(funnel.share_of_raw.iloc[-1]), "url_blocked": url_drops, "non_english": lang_drops,
        "exact_duplicates": s["exact_deduped"]["docs_in"] - s["exact_deduped"]["docs_out"],
        "quality_rejected": s["quality_filtered"]["docs_in"] - s["quality_filtered"]["docs_out"],
        "near_duplicates": s["near_deduped"]["docs_in"] - s["near_deduped"]["docs_out"],
        "near_dup_clusters": s["near_deduped"]["clusters_with_duplicates"], "largest_cluster": s["near_deduped"]["largest_cluster"],
        "contaminated_removed": s["decontaminated_redacted"]["docs_in"] - s["decontaminated_redacted"]["docs_out"],
        "contamination_by_benchmark": contam.to_dict(),
        "pii_redactions": {k.replace("pii_", ""): v for k, v in s["decontaminated_redacted"].items() if k.startswith("pii_")},
        "docs_with_pii": int((corpus.pii_redactions > 0).sum()),
        "classifier": s["quality_classifier"], "hq_documents": s["scored_corpus"]["hq_docs"],
        "tokens_all": s["token_shards"]["tokens_all"], "tokens_hq": s["token_shards"]["tokens_hq"],
        "words": s["scored_corpus"]["words"], "pipeline_wall_seconds": float(timing.seconds.sum() + s["quality_classifier"]["seconds"]),
        "cc_cld2_agrees_english": float(corpus.cc_language.str.startswith("eng").mean()),
    }
    (C.REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    md = lambda f: f.to_markdown(index=False, floatfmt=",.3f")  # noqa: E731
    lines = [
        "# Results (auto-generated by scripts/run_pipeline.py)\n",
        "## Funnel\n", md(funnel),
        "\n## Rejections by stage and reason (top 25)\n", md(pd.read_csv(C.TABLES / "drop_reasons.csv").head(25)),
        "\n## Stage timing\n", md(timing),
        f"\n## Quality classifier\n\n```json\n{json.dumps(s['quality_classifier'], indent=2)}\n```\n",
        f"\n## All metrics\n\n```json\n{json.dumps(metrics, indent=2, default=float)}\n```\n",
    ]
    (C.REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    _dataset_card(metrics)
    return metrics


def _dataset_card(m: dict) -> None:
    card = f"""# Dataset card: refined Common Crawl sample ({m['crawl']})

| | |
|---|---|
| Source | {m['wet_files']} WET files from Common Crawl {m['crawl']}, sampled evenly across the crawl |
| Documents | {m['final_documents']:,} ({100 * m['final_share_of_raw']:.1f}% of {m['raw_documents']:,} raw pages) |
| Tokens (o200k_base) | {m['tokens_all']:,} all; {m['tokens_hq']:,} in the high-quality subset |
| Language | English (fastText lid.176, P ≥ {C.LANG_THRESHOLD}) |
| Format | Parquet (`corpus/`), uint32 token shards with manifests (`tokens/`) |

## Processing
URL blocklist → language ID → exact dedup → C4 line cleaning, Gopher repetition and quality rules, FineWeb filters →
MinHash LSH near-dedup (word 5-grams, 14 bands × 8 rows) → 13-gram benchmark decontamination (MMLU, ARC-Challenge, GSM8K,
TruthfulQA, HumanEval) → PII redaction (e-mail, phone, public IP, Luhn-valid card numbers) → fastText quality score.

## Fields
`doc_id`, `url`, `domain`, `date`, `cc_language`, `lang_score`, `text`, `cluster_size` (near-duplicates collapsed into this document),
`pii_redactions`, `quality_score`, `high_quality` (top {int(C.HQ_TOP_FRACTION * 100)}%), `n_words`.

## Known limitations
- PII redaction is pattern-based: names and street addresses are not removed.
- The URL blocklist is short; FineWeb uses the full UT1 list.
- Quality scores reflect similarity to Wikipedia lead sections, which favours encyclopedic prose over other good writing.
- Web text carries the biases of its sources. Licensing follows Common Crawl's terms of use.
"""
    (C.OUTPUT / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    (C.REPORTS / "DATASET_CARD.md").write_text(card, encoding="utf-8")
