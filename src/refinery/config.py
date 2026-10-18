"""Paths and every threshold the refinery uses, in one place.

Thresholds follow published pipelines so each choice has a citation:
Gopher (Rae et al., 2021), C4 (Raffel et al., 2020), FineWeb (Penedo et al., 2024).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
WET_DIR = RAW / "wet"
BENCH_DIR = RAW / "benchmarks"
LID_MODEL = RAW / "lid.176.ftz"
STAGES = ROOT / "data" / "stages"
OUTPUT = ROOT / "data" / "output"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

SEED = 13
WORKERS = 8

# ---- language identification (fastText lid.176, as FineWeb) ----
LANGUAGE = "en"
LANG_THRESHOLD = 0.65

# ---- Gopher quality filter ----
GOPHER_MIN_WORDS, GOPHER_MAX_WORDS = 50, 100_000
GOPHER_MEAN_WORD_LEN = (3, 10)
GOPHER_MAX_SYMBOL_WORD_RATIO = 0.1          # "#" and "..." per word
GOPHER_MAX_BULLET_LINES = 0.9
GOPHER_MAX_ELLIPSIS_LINES = 0.3
GOPHER_MIN_ALPHA_WORDS = 0.8
GOPHER_STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
GOPHER_MIN_STOP_WORDS = 2

# ---- Gopher repetition filter ----
GOPHER_DUP_LINE_FRAC = 0.30
GOPHER_DUP_PARA_FRAC = 0.30
GOPHER_DUP_LINE_CHAR_FRAC = 0.20
GOPHER_DUP_PARA_CHAR_FRAC = 0.20
GOPHER_TOP_NGRAM_CHAR_FRAC = {2: 0.20, 3: 0.18, 4: 0.16}
GOPHER_DUP_NGRAM_CHAR_FRAC = {5: 0.15, 6: 0.14, 7: 0.13, 8: 0.12, 9: 0.11, 10: 0.10}

# ---- C4 line-level cleaning ----
C4_MIN_WORDS_PER_LINE = 3
C4_BAD_LINE_SUBSTRINGS = ("javascript", "lorem ipsum", "terms of use", "privacy policy",
                          "cookie policy", "uses cookies", "use of cookies", "all rights reserved")

# ---- FineWeb custom filters ----
FINEWEB_MIN_PUNCT_LINE_FRAC = 0.12
FINEWEB_MAX_DUP_LINE_CHAR_FRAC = 0.10
FINEWEB_MAX_SHORT_LINE_FRAC = 0.67          # lines under 30 characters

# ---- near-duplicate detection (FineWeb settings) ----
MINHASH_NGRAM = 5
MINHASH_BANDS, MINHASH_ROWS = 14, 8         # 112 hash functions
MINHASH_VERIFY_JACCARD = 0.70               # candidate pairs below this estimated Jaccard are not merged

# ---- benchmark decontamination (GPT-3 style 13-gram overlap) ----
DECONTAM_NGRAM = 13

# ---- quality classifier and tokenisation ----
CLASSIFIER_WORDS = 300                      # both classes truncated to the same length
WIKI_POSITIVES = 2000
HQ_TOP_FRACTION = 0.30                      # "high-quality" subset = top 30% by classifier score
TOKENIZER = "o200k_base"
SHARD_TOKENS = 20_000_000
