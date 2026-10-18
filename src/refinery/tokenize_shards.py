"""Tokenise the refined corpus into fixed-size training shards.

Documents are encoded with tiktoken's o200k_base vocabulary and joined with an
end-of-text token, then cut into shards of 20M tokens saved as uint32 .npy
files, the format most training loops memory-map directly. A manifest records
each shard's token count and checksum so a trainer can verify what it reads.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken

from . import config as C


def encoder():
    return tiktoken.get_encoding(C.TOKENIZER)


def write_shards(texts: list[str], out_dir: Path, prefix: str, shard_tokens: int = C.SHARD_TOKENS) -> dict:
    enc = encoder()
    eot = enc.eot_token
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{prefix}-*.npy"):
        old.unlink()  # idempotent re-runs
    shards, buffer, total, doc_lengths = [], [], 0, []
    batch = 2000
    for start in range(0, len(texts), batch):
        for tokens in enc.encode_ordinary_batch(texts[start:start + batch], num_threads=8):
            tokens.append(eot)
            doc_lengths.append(len(tokens))
            buffer.extend(tokens)
            while len(buffer) >= shard_tokens:
                shards.append(_flush(buffer[:shard_tokens], out_dir, prefix, len(shards)))
                buffer = buffer[shard_tokens:]
    if buffer:
        shards.append(_flush(buffer, out_dir, prefix, len(shards)))
    total = sum(s["tokens"] for s in shards)
    manifest = {"tokenizer": C.TOKENIZER, "vocab_size": enc.n_vocab, "eot_token": eot, "documents": len(texts),
                "tokens": total, "mean_tokens_per_doc": float(np.mean(doc_lengths)) if doc_lengths else 0.0,
                "shards": shards}
    (out_dir / f"{prefix}-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _flush(tokens: list[int], out_dir: Path, prefix: str, index: int) -> dict:
    arr = np.asarray(tokens, dtype=np.uint32)
    path = out_dir / f"{prefix}-{index:05d}.npy"
    np.save(path, arr)
    return {"file": path.name, "tokens": int(arr.size), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
