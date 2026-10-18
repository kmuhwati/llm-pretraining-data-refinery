"""The refinery's stages as plain functions over per-file Parquet.

Per-document work (extraction, language ID, quality filters, MinHash,
decontamination, PII) runs file-by-file in a process pool. Global work (exact
and near dedup) only ever holds hashes and signatures, never document text, so
memory stays flat as the corpus grows. Every stage writes its survivors to
data/stages/<stage>/<file>.parquet and every rejection, with its reason, to
data/stages/drops/<stage>/<file>.parquet. Re-running a stage overwrites its own
outputs only, so the pipeline is idempotent.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xxhash

from . import config as C
from . import filters
from .warc import iter_wet

STAGE_DIRS = {
    "extract": "01_extract", "exact": "02_exact_dedup", "quality": "03_quality",
    "signatures": "04_signatures", "near": "05_near_dedup", "clean": "06_decontam_pii",
}


def stage_dir(stage: str) -> Path:
    path = C.STAGES / STAGE_DIRS[stage]
    path.mkdir(parents=True, exist_ok=True)
    return path


def drops_dir(stage: str) -> Path:
    path = C.STAGES / "drops" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def wet_files() -> list[Path]:
    files = sorted(C.WET_DIR.glob("*.warc.wet.gz"))
    assert len({file_key(p) for p in files}) == len(files), "two WET files map to the same stage key"
    return files


def file_key(path: Path) -> str:
    parts = path.name.split(".")[0].split("-")    # CC-MAIN-<segment start>-<segment end>-<index>
    return f"{parts[2]}-{parts[-1]}"              # the index alone repeats across segments


def _write(frame: pd.DataFrame, folder: Path, key: str) -> None:
    frame.to_parquet(folder / f"{key}.parquet", index=False)


def _write_drops(rows: list[dict], stage: str, key: str) -> None:
    pd.DataFrame(rows, columns=["doc_id", "reason"]).to_parquet(drops_dir(stage) / f"{key}.parquet", index=False)


def _pool_map(fn, items, workers=C.WORKERS):
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


# ------------------------------------------------------------------ 1. extract + URL + language
_LID = None


def _lid():
    global _LID
    if _LID is None:
        import fasttext
        fasttext.FastText.eprint = lambda *a, **k: None  # silence a deprecation print
        _LID = fasttext.load_model(str(C.LID_MODEL))
    return _LID


def fasttext_top1(model, text: str) -> tuple[str, float]:
    """Top label and probability. Calls the C++ predictor directly: the Python
    wrapper in fasttext-wheel uses a NumPy 1.x-only call that fails on NumPy 2."""
    preds = model.f.predict(text.replace("\n", " "), 1, 0.0, "replace")
    if not preds:
        return "unknown", 0.0
    prob, label = preds[0]
    return label.replace("__label__", ""), float(prob)


def _extract_one(path: Path) -> dict:
    key = file_key(path)
    t0 = time.time()
    keep, drops, n, raw_chars = [], [], 0, 0
    model = _lid()
    for seq, rec in enumerate(iter_wet(path)):
        n += 1
        raw_chars += len(rec.text)
        doc_id = f"{key}-{seq:06d}"
        reason = filters.url_filter(rec.url)
        if reason:
            drops.append({"doc_id": doc_id, "reason": reason})
            continue
        lang, score = fasttext_top1(model, rec.text[:5000])
        if lang != C.LANGUAGE or score < C.LANG_THRESHOLD:
            drops.append({"doc_id": doc_id, "reason": f"lang:{lang if lang != C.LANGUAGE else 'low_confidence'}"})
            continue
        keep.append({"doc_id": doc_id, "url": rec.url, "domain": (urlparse(rec.url).hostname or "").lower(),
                     "date": rec.date, "cc_language": rec.cc_language, "lang_score": score, "text": rec.text})
    _write(pd.DataFrame(keep), stage_dir("extract"), key)
    _write_drops(drops, "extract", key)
    return {"file": key, "docs_in": n, "docs_out": len(keep), "raw_chars": raw_chars, "seconds": time.time() - t0}


def extract(files: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(_pool_map(_extract_one, files))


# ------------------------------------------------------------------ 2. exact dedup (global)
def exact_dedup(files: list[Path]) -> dict:
    seen: set[int] = set()
    n_in = n_out = 0
    for path in files:
        key = file_key(path)
        df = pd.read_parquet(stage_dir("extract") / f"{key}.parquet")
        hashes = [xxhash.xxh64_intdigest(filters.normalize_for_hash(t).encode()) for t in df.text]
        mask = []
        for h in hashes:
            mask.append(h not in seen)
            seen.add(h)
        mask = np.array(mask, dtype=bool)
        n_in += len(df)
        n_out += int(mask.sum())
        _write(df[mask], stage_dir("exact"), key)
        _write_drops([{"doc_id": d, "reason": "exact_duplicate"} for d in df.doc_id[~mask]], "exact", key)
    return {"docs_in": n_in, "docs_out": n_out}


# ------------------------------------------------------------------ 3. quality filters
def _quality_one(key: str) -> dict:
    df = pd.read_parquet(stage_dir("exact") / f"{key}.parquet")
    keep, drops = [], []
    for row in df.itertuples(index=False):
        cleaned, reason = filters.quality_pipeline(row.text)
        if reason:
            drops.append({"doc_id": row.doc_id, "reason": reason})
        else:
            keep.append({**row._asdict(), "text": cleaned})
    _write(pd.DataFrame(keep, columns=df.columns), stage_dir("quality"), key)
    _write_drops(drops, "quality", key)
    return {"file": key, "docs_in": len(df), "docs_out": len(keep)}


def quality(files: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(_pool_map(_quality_one, [file_key(p) for p in files]))


# ------------------------------------------------------------------ 4. MinHash signatures
def _signatures_one(key: str) -> int:
    from .minhash import MinHasher

    hasher = MinHasher()
    df = pd.read_parquet(stage_dir("quality") / f"{key}.parquet", columns=["doc_id", "text"])
    sigs = np.stack([hasher.signature(t) for t in df.text]) if len(df) else np.zeros((0, hasher.num_perm), np.uint32)
    np.savez(stage_dir("signatures") / f"{key}.npz", ids=df.doc_id.to_numpy(), sigs=sigs)
    return len(df)


def signatures(files: list[Path]) -> int:
    return sum(_pool_map(_signatures_one, [file_key(p) for p in files]))


# ------------------------------------------------------------------ 5. near dedup (global LSH)
def near_dedup(files: list[Path]) -> dict:
    from .minhash import lsh_clusters

    ids, sigs = [], []
    for path in files:
        z = np.load(stage_dir("signatures") / f"{file_key(path)}.npz", allow_pickle=True)
        ids.append(z["ids"])
        sigs.append(z["sigs"])
    ids = np.concatenate(ids)
    roots, stats = lsh_clusters(np.concatenate(sigs))
    cluster_size = pd.Series(roots).map(pd.Series(roots).value_counts()).to_numpy()
    keep = roots == np.arange(len(roots))  # the earliest document of each cluster survives
    info = pd.DataFrame({"doc_id": ids, "keep": keep, "cluster_size": cluster_size})
    clusters = pd.Series(cluster_size[keep])
    for path in files:
        key = file_key(path)
        df = pd.read_parquet(stage_dir("quality") / f"{key}.parquet").merge(info, on="doc_id")
        _write(df[df.keep].drop(columns="keep"), stage_dir("near"), key)
        _write_drops([{"doc_id": d, "reason": "near_duplicate"} for d in df.doc_id[~df.keep]], "near", key)
    return {"docs_in": int(len(ids)), "docs_out": int(keep.sum()), **stats,
            "clusters_with_duplicates": int((clusters > 1).sum()), "largest_cluster": int(clusters.max()) if len(clusters) else 0}


# ------------------------------------------------------------------ 6. decontamination + PII
_DECON = None


def _clean_one(key: str) -> dict:
    global _DECON
    from .decontam import Decontaminator
    from .pii import redact

    if _DECON is None:
        _DECON = Decontaminator()
    df = pd.read_parquet(stage_dir("near") / f"{key}.parquet")
    keep, drops = [], []
    pii_totals = {"email": 0, "phone": 0, "ip": 0, "card": 0}
    for row in df.itertuples(index=False):
        bench = _DECON.match(row.text)
        if bench:
            drops.append({"doc_id": row.doc_id, "reason": f"contaminated:{bench}"})
            continue
        text, counts = redact(row.text)
        for k, v in counts.items():
            pii_totals[k] += v
        keep.append({**row._asdict(), "text": text, "pii_redactions": sum(counts.values())})
    _write(pd.DataFrame(keep), stage_dir("clean"), key)
    _write_drops(drops, "clean", key)
    return {"file": key, "docs_in": len(df), "docs_out": len(keep), **{f"pii_{k}": v for k, v in pii_totals.items()}}


def decontaminate_and_redact(files: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(_pool_map(_clean_one, [file_key(p) for p in files], workers=min(C.WORKERS, 6)))


def read_stage(stage: str, files: list[Path], columns=None) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(stage_dir(stage) / f"{file_key(p)}.parquet", columns=columns) for p in files],
                     ignore_index=True)


def read_drops(stage: str) -> pd.DataFrame:
    parts = [pd.read_parquet(p).assign(stage=stage) for p in sorted(drops_dir(stage).glob("*.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["doc_id", "reason", "stage"])


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
