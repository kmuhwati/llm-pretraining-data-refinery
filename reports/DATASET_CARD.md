# Dataset card: refined Common Crawl sample (CC-MAIN-2026-39)

| | |
|---|---|
| Source | 16 WET files from Common Crawl CC-MAIN-2026-39, sampled evenly across the crawl |
| Documents | 46,552 (13.7% of 339,259 raw pages) |
| Tokens (o200k_base) | 62,057,879 all; 21,877,944 in the high-quality subset |
| Language | English (fastText lid.176, P ≥ 0.65) |
| Format | Parquet (`corpus/`), uint32 token shards with manifests (`tokens/`) |

## Processing
URL blocklist → language ID → exact dedup → C4 line cleaning, Gopher repetition and quality rules, FineWeb filters →
MinHash LSH near-dedup (word 5-grams, 14 bands × 8 rows) → 13-gram benchmark decontamination (MMLU, ARC-Challenge, GSM8K,
TruthfulQA, HumanEval) → PII redaction (e-mail, phone, public IP, Luhn-valid card numbers) → fastText quality score.

## Fields
`doc_id`, `url`, `domain`, `date`, `cc_language`, `lang_score`, `text`, `cluster_size` (near-duplicates collapsed into this document),
`pii_redactions`, `quality_score`, `high_quality` (top 30%), `n_words`.

## Known limitations
- PII redaction is pattern-based: names and street addresses are not removed.
- The URL blocklist is short; FineWeb uses the full UT1 list.
- Quality scores reflect similarity to Wikipedia lead sections, which favours encyclopedic prose over other good writing.
- Web text carries the biases of its sources. Licensing follows Common Crawl's terms of use.
