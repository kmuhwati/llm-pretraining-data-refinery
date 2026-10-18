"""Unit tests on synthetic inputs with known answers (no crawl data needed)."""
import gzip
import io
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refinery import filters, pii  # noqa: E402
from refinery.decontam import Decontaminator  # noqa: E402
from refinery.minhash import MinHasher, jaccard_estimate, lsh_clusters  # noqa: E402
from refinery.warc import iter_wet  # noqa: E402

rng = random.Random(0)
VOCAB = [f"w{i}" for i in range(5000)]
PROSE = ("The committee met on Tuesday to review the budget. Members discussed the new library and the cost of "
         "repairs to the old bridge, which has been closed since spring. After a long debate, they agreed to "
         "fund both projects over two years. The mayor said that the town would publish the full plan online "
         "and that residents could comment on it until the end of the month. Several speakers asked for more "
         "detail about how the money would be raised, and the treasurer promised a written answer.")


def random_doc(n=400):
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def true_jaccard(a, b, n=5):
    grams = lambda t: {" ".join(t.split()[i:i + n]) for i in range(len(t.split()) - n + 1)}  # noqa: E731
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / len(ga | gb)


# ------------------------------------------------------------------ MinHash / LSH
def test_minhash_estimates_jaccard():
    h = MinHasher(num_perm=256)
    base = random_doc(600).split()
    for keep in (0.95, 0.8, 0.5):
        other = base[:]
        for i in range(int(len(other) * (1 - keep))):
            other[rng.randrange(len(other))] = rng.choice(VOCAB)
        a, b = " ".join(base), " ".join(other)
        assert abs(jaccard_estimate(h.signature(a), h.signature(b)) - true_jaccard(a, b)) < 0.08


def test_lsh_groups_near_duplicates_and_keeps_unique_docs_apart():
    h = MinHasher()
    originals = [random_doc() for _ in range(30)]
    near_copies = []
    for doc in originals[:10]:
        words = doc.split()
        words[rng.randrange(len(words))] = "edited"
        near_copies.append(" ".join(words))
    docs = originals + near_copies
    roots, _ = lsh_clusters(np.stack([h.signature(d) for d in docs]))
    for i in range(10):
        assert roots[30 + i] == roots[i]          # each near copy joins its original
    assert len(set(roots[10:30])) == 20           # untouched documents stay separate


# ------------------------------------------------------------------ quality filters
def test_real_prose_passes_every_filter():
    cleaned, reason = filters.quality_pipeline(PROSE)
    assert reason is None and "committee" in cleaned


def test_repeated_lines_fail_gopher_repetition():
    text = "\n".join(["Buy now and save big on every order today."] * 20 + [PROSE])
    assert filters.gopher_repetition(text).startswith("gopher_rep")


@pytest.mark.parametrize("text,reason", [
    ("Too short to be a document.", "gopher_q:too_short"),
    ("#news #today " * 20 + PROSE, "gopher_q:symbol_ratio"),
])
def test_gopher_quality_rules(text, reason):
    assert filters.gopher_quality(text) == reason


def test_c4_drops_boilerplate_lines_and_code_pages():
    cleaned, reason = filters.c4_clean(PROSE + "\nThis site uses cookies to improve your experience.\nHome\n")
    assert reason is None and "cookies" not in cleaned and "Home" not in cleaned
    assert filters.c4_clean("function f() { return 1; } " + PROSE)[1] == "c4:curly_brace"
    assert filters.c4_clean("Lorem ipsum dolor sit amet. " + PROSE)[1] == "c4:lorem_ipsum"


def test_url_blocklist():
    assert filters.url_filter("https://example-casino.com/page") == "url:blocklist"
    assert filters.url_filter("https://www.bbc.co.uk/news") is None


# ------------------------------------------------------------------ PII
def test_pii_redaction():
    text = ("Mail jane.doe@example.org or call +1 415-555-0132. Server 8.8.8.8, router 192.168.1.1. "
            "Card 4111 1111 1111 1111, order 1234567890123.")
    out, counts = pii.redact(text)
    assert "<EMAIL>" in out and "<PHONE>" in out and "<IP>" in out and "<CARD>" in out
    assert "192.168.1.1" in out                 # private IPs are not personal data
    assert "1234567890123" in out               # fails the Luhn check, so it is not a card
    assert counts == {"email": 1, "phone": 1, "ip": 1, "card": 1}


# ------------------------------------------------------------------ decontamination
def test_decontamination_catches_planted_benchmark_text():
    question = "Natalia sold clips to 48 of her friends in April and then she sold half as many clips in May how many did she sell"
    d = Decontaminator({"GSM8K": [question]})
    assert d.match(PROSE + " " + question) == "GSM8K"
    assert d.match(PROSE) is None


# ------------------------------------------------------------------ WET parsing
def test_wet_reader_parses_conversion_records(tmp_path):
    body = "Hello world from a web page.".encode()
    record = (b"WARC/1.0\r\nWARC-Type: conversion\r\nWARC-Target-URI: https://example.com/a\r\n"
              b"WARC-Date: 2026-09-04T13:16:03Z\r\nWARC-Record-ID: <urn:uuid:1>\r\n"
              b"WARC-Identified-Content-Language: eng\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body + b"\r\n\r\n")
    info = b"WARC/1.0\r\nWARC-Type: warcinfo\r\nContent-Length: 4\r\n\r\ninfo\r\n\r\n"
    path = tmp_path / "x.warc.wet.gz"
    path.write_bytes(gzip.compress(info + record))
    records = list(iter_wet(path))
    assert len(records) == 1 and records[0].url == "https://example.com/a" and records[0].text == body.decode()


# ------------------------------------------------------------------ tokenised shards
def test_token_shards_are_complete_and_deterministic(tmp_path):
    from refinery.tokenize_shards import write_shards

    texts = [PROSE, "A second short document.", "Third."]
    m1 = write_shards(texts, tmp_path, "t", shard_tokens=50)
    arrays = [np.load(tmp_path / s["file"]) for s in m1["shards"]]
    tokens = np.concatenate(arrays)
    assert tokens.size == m1["tokens"] and (tokens == m1["eot_token"]).sum() == len(texts)
    m2 = write_shards(texts, tmp_path, "t", shard_tokens=50)
    assert [s["sha256"] for s in m1["shards"]] == [s["sha256"] for s in m2["shards"]]
