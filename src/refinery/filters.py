"""Document filters. Each returns None to keep a document or a short reason string.

Implementations follow the published rules:
- URL filter: a small adult/spam blocklist (FineWeb uses the larger UT1 list).
- C4 (Raffel et al., 2020): drop boilerplate lines, then reject pages with code
  braces or "lorem ipsum".
- Gopher repetition and quality (Rae et al., 2021, Table A1).
- FineWeb custom filters (Penedo et al., 2024).
"""
from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urlparse

from . import config as C

BLOCKED_URL_TERMS = (
    "porn", "xxx", "sex", "escort", "nude", "hentai", "camgirl", "onlyfans", "casino", "betting",
    "viagra", "cialis", "replica-watches", "payday-loan",
)
TERMINAL_PUNCT = (".", "!", "?", '"', "'", "”", "’")
BULLETS = ("•", "-", "*", "‣", "▪", "●")


# ------------------------------------------------------------------ URL
def url_filter(url: str) -> str | None:
    host = (urlparse(url).hostname or "").lower()
    path = url.lower()
    if not host:
        return "url:invalid"
    if any(term in host or term in path for term in BLOCKED_URL_TERMS):
        return "url:blocklist"
    return None


# ------------------------------------------------------------------ C4
def c4_clean(text: str) -> tuple[str, str | None]:
    """Remove boilerplate lines; reject pages C4 discards outright."""
    lower = text.lower()
    if "lorem ipsum" in lower:
        return text, "c4:lorem_ipsum"
    if "{" in text:
        return text, "c4:curly_brace"
    kept = []
    for line in text.split("\n"):
        line_lower = line.lower()
        if any(bad in line_lower for bad in C.C4_BAD_LINE_SUBSTRINGS):
            continue
        if line.strip() and len(line.split()) < C.C4_MIN_WORDS_PER_LINE and not line.rstrip().endswith(TERMINAL_PUNCT):
            continue  # navigation crumbs, button labels, menu items
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    if not cleaned:
        return cleaned, "c4:empty_after_cleaning"
    return cleaned, None


# ------------------------------------------------------------------ Gopher repetition
def _dup_fraction(items: list[str]) -> tuple[float, float]:
    """(fraction of items that repeat an earlier item, fraction of characters in them)."""
    seen, dup_items, dup_chars, total_chars = set(), 0, 0, 0
    for item in items:
        total_chars += len(item)
        if item in seen:
            dup_items += 1
            dup_chars += len(item)
        else:
            seen.add(item)
    n = max(len(items), 1)
    return dup_items / n, dup_chars / max(total_chars, 1)


def _top_ngram_char_frac(words: list[str], n: int) -> float:
    grams = Counter(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    if not grams:
        return 0.0
    gram, count = grams.most_common(1)[0]
    return len(gram) * count / max(sum(len(w) for w in words), 1)


def _dup_ngram_char_frac(words: list[str], n: int) -> float:
    """Characters covered by n-grams already seen earlier (non-overlapping), over all characters."""
    seen, covered, i = set(), 0, 0
    total = sum(len(w) for w in words)
    while i <= len(words) - n:
        gram = tuple(words[i:i + n])
        if gram in seen:
            covered += sum(len(w) for w in gram)
            i += n
        else:
            seen.add(gram)
            i += 1
    return covered / max(total, 1)


def gopher_repetition(text: str) -> str | None:
    lines = [line for line in text.split("\n") if line.strip()]
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    line_frac, line_char_frac = _dup_fraction(lines)
    para_frac, para_char_frac = _dup_fraction(paragraphs)
    if line_frac > C.GOPHER_DUP_LINE_FRAC:
        return "gopher_rep:dup_lines"
    if para_frac > C.GOPHER_DUP_PARA_FRAC:
        return "gopher_rep:dup_paragraphs"
    if line_char_frac > C.GOPHER_DUP_LINE_CHAR_FRAC:
        return "gopher_rep:dup_line_chars"
    if para_char_frac > C.GOPHER_DUP_PARA_CHAR_FRAC:
        return "gopher_rep:dup_paragraph_chars"
    words = text.split()
    for n, limit in C.GOPHER_TOP_NGRAM_CHAR_FRAC.items():
        if _top_ngram_char_frac(words, n) > limit:
            return f"gopher_rep:top_{n}gram"
    for n, limit in C.GOPHER_DUP_NGRAM_CHAR_FRAC.items():
        if _dup_ngram_char_frac(words, n) > limit:
            return f"gopher_rep:dup_{n}gram"
    return None


# ------------------------------------------------------------------ Gopher quality
def gopher_quality(text: str) -> str | None:
    words = text.split()
    n = len(words)
    if n < C.GOPHER_MIN_WORDS:
        return "gopher_q:too_short"
    if n > C.GOPHER_MAX_WORDS:
        return "gopher_q:too_long"
    mean_len = sum(len(w) for w in words) / n
    if not C.GOPHER_MEAN_WORD_LEN[0] <= mean_len <= C.GOPHER_MEAN_WORD_LEN[1]:
        return "gopher_q:mean_word_length"
    symbols = text.count("#") + text.count("...") + text.count("…")
    if symbols / n > C.GOPHER_MAX_SYMBOL_WORD_RATIO:
        return "gopher_q:symbol_ratio"
    lines = [line for line in text.split("\n") if line.strip()]
    if lines:
        if sum(line.lstrip().startswith(BULLETS) for line in lines) / len(lines) > C.GOPHER_MAX_BULLET_LINES:
            return "gopher_q:bullet_lines"
        if sum(line.rstrip().endswith(("...", "…")) for line in lines) / len(lines) > C.GOPHER_MAX_ELLIPSIS_LINES:
            return "gopher_q:ellipsis_lines"
    if sum(any(ch.isalpha() for ch in w) for w in words) / n < C.GOPHER_MIN_ALPHA_WORDS:
        return "gopher_q:alpha_words"
    lowered = {w.lower().strip(".,;:!?\"'()") for w in words}
    if len(lowered & C.GOPHER_STOP_WORDS) < C.GOPHER_MIN_STOP_WORDS:
        return "gopher_q:stop_words"
    return None


# ------------------------------------------------------------------ FineWeb
def fineweb_filters(text: str) -> str | None:
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return "fineweb:empty"
    if sum(line.rstrip().endswith(TERMINAL_PUNCT) for line in lines) / len(lines) < C.FINEWEB_MIN_PUNCT_LINE_FRAC:
        return "fineweb:line_punctuation"
    _, dup_char_frac = _dup_fraction(lines)
    if dup_char_frac > C.FINEWEB_MAX_DUP_LINE_CHAR_FRAC:
        return "fineweb:dup_line_chars"
    if sum(len(line) < 30 for line in lines) / len(lines) > C.FINEWEB_MAX_SHORT_LINE_FRAC:
        return "fineweb:short_lines"
    return None


def quality_pipeline(text: str) -> tuple[str, str | None]:
    """C4 cleaning, then Gopher repetition, Gopher quality and FineWeb filters, in that order."""
    cleaned, reason = c4_clean(text)
    if reason:
        return cleaned, reason
    for check in (gopher_repetition, gopher_quality, fineweb_filters):
        reason = check(cleaned)
        if reason:
            return cleaned, reason
    return cleaned, None


def normalize_for_hash(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation: exact-duplicate key."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()
