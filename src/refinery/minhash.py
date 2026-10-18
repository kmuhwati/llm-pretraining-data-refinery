"""Vectorised MinHash and LSH near-duplicate clustering.

Each document becomes a set of word 5-grams. MinHash compresses that set into
112 numbers whose agreement rate between two documents estimates their Jaccard
similarity. LSH splits the 112 numbers into 14 bands of 8: two documents become
candidates if any band matches exactly, which happens with high probability
only above roughly 0.72 Jaccard ((1/14)^(1/8)). Candidates are verified on the
full signature, then grouped with union-find.

Hash family: multiply-shift on 64-bit words, h(x) = ((a*x + b) mod 2^64) >> 32,
with odd `a`. numpy's uint64 arithmetic wraps, which is exactly the mod 2^64.
"""
from __future__ import annotations

import re

import numpy as np
import xxhash

from . import config as C

_WORD = re.compile(r"\w+", re.UNICODE)


class MinHasher:
    def __init__(self, num_perm: int = C.MINHASH_BANDS * C.MINHASH_ROWS, ngram: int = C.MINHASH_NGRAM, seed: int = C.SEED):
        rng = np.random.default_rng(seed)
        self.a = (rng.integers(1, 2**63, num_perm, dtype=np.uint64) << np.uint64(1)) | np.uint64(1)  # odd
        self.b = rng.integers(0, 2**63, num_perm, dtype=np.uint64)
        self.ngram, self.num_perm = ngram, num_perm

    def shingles(self, text: str) -> np.ndarray:
        words = _WORD.findall(text.lower())
        n = self.ngram
        if len(words) < n:
            grams = [" ".join(words)] if words else [""]
        else:
            grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        return np.fromiter((xxhash.xxh64_intdigest(g.encode()) for g in set(grams)), dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        x = self.shingles(text)
        with np.errstate(over="ignore"):
            hashed = (self.a[:, None] * x[None, :] + self.b[:, None]) >> np.uint64(32)
        return hashed.min(axis=1).astype(np.uint32)


def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)  # the earliest document becomes the cluster root


def lsh_clusters(signatures: np.ndarray, bands: int = C.MINHASH_BANDS, rows: int = C.MINHASH_ROWS,
                 verify: float = C.MINHASH_VERIFY_JACCARD) -> tuple[np.ndarray, dict]:
    """Return the cluster root for every document plus candidate/merge counts."""
    n = len(signatures)
    uf = UnionFind(n)
    candidates = merged = 0
    for band in range(bands):
        block = np.ascontiguousarray(signatures[:, band * rows:(band + 1) * rows])
        keys = [xxhash.xxh64_intdigest(row.tobytes()) for row in block]
        buckets: dict[int, int] = {}
        for i, key in enumerate(keys):
            first = buckets.setdefault(key, i)
            if first != i:
                candidates += 1
                if uf.find(first) != uf.find(i) and jaccard_estimate(signatures[first], signatures[i]) >= verify:
                    uf.union(first, i)
                    merged += 1
    roots = np.array([uf.find(i) for i in range(n)])
    return roots, {"candidate_pairs": candidates, "merges": merged}
