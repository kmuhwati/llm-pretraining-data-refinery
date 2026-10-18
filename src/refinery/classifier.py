"""Model-based quality scoring (the GPT-3 / DCLM idea, at laptop scale).

A fastText classifier learns to tell reference-quality prose (Wikipedia lead
sections) from random English web pages that passed language ID but no other
filter. Its probability becomes `quality_score` on every surviving document,
and the top 30% form the "high-quality" subset. Both classes are truncated to
the same number of words so the model cannot simply learn document length.
"""
from __future__ import annotations

import json
import random
import re
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from . import config as C

WIKI_CACHE = C.RAW / "wikipedia_leads.jsonl"
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "kmuhwati-llm-data-refinery/1.0 (https://github.com/kmuhwati)"}


def fetch_wikipedia_leads(n: int = C.WIKI_POSITIVES, min_words: int = 80) -> list[str]:
    """Random article lead sections, 20 per API call, cached on disk."""
    cached = [json.loads(line)["text"] for line in WIKI_CACHE.read_text(encoding="utf-8").splitlines()] if WIKI_CACHE.exists() else []
    seen = set(cached)
    with httpx.Client(headers=UA, timeout=30) as client:
        while len(cached) < n:
            r = client.get(API, params={"action": "query", "format": "json", "generator": "random", "grnnamespace": 0,
                                        "grnlimit": 20, "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 20})
            for page in r.json().get("query", {}).get("pages", {}).values():
                text = page.get("extract", "").strip()
                if len(text.split()) >= min_words and text not in seen:
                    cached.append(text)
                    seen.add(text)
            time.sleep(0.5)  # be polite to the API
    WIKI_CACHE.write_text("\n".join(json.dumps({"text": t}) for t in cached[:n]), encoding="utf-8")
    return cached[:n]


def _prep(text: str) -> str:
    words = re.sub(r"\s+", " ", text).split(" ")[: C.CLASSIFIER_WORDS]
    return " ".join(words).lower()


def _predict_hq(model, text: str) -> float:
    preds = dict((label, prob) for prob, label in model.f.predict(_prep(text), 2, 0.0, "replace"))
    return float(preds.get("__label__hq", 1 - preds.get("__label__web", 0.0)))


class QualityClassifier:
    def __init__(self, model):
        self.model = model

    @classmethod
    def train(cls, positives: list[str], negatives: list[str], seed: int = C.SEED):
        import fasttext

        rng = random.Random(seed)
        rows = [("hq", t) for t in positives] + [("web", t) for t in negatives]
        rng.shuffle(rows)
        cut = int(len(rows) * 0.8)
        train, test = rows[:cut], rows[cut:]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.txt"
            path.write_text("\n".join(f"__label__{y} {_prep(t)}" for y, t in train), encoding="utf-8")
            model = fasttext.train_supervised(str(path), epoch=15, lr=0.5, wordNgrams=2, dim=64, bucket=200_000, seed=seed,
                                          verbose=0, thread=1)  # 200k n-gram buckets: ample for 4k docs, 10x less memory
        clf = cls(model)
        y = np.array([1 if label == "hq" else 0 for label, _ in test])
        p = np.array([clf.score(t) for _, t in test])
        metrics = {"train_docs": len(train), "test_docs": len(test), "test_accuracy": float(accuracy_score(y, p >= 0.5)),
                   "test_roc_auc": float(roc_auc_score(y, p))}
        return clf, metrics

    def score(self, text: str) -> float:
        return _predict_hq(self.model, text)

    def save(self, path: Path) -> None:
        self.model.save_model(str(path))
