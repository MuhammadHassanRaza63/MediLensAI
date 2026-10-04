"""Small retrieval layer over curated medical notes.

MVP: keyed lookup (test + direction) plus a lightweight keyword score over any
ingested MedlinePlus text. The interface (``KnowledgeBase.retrieve``) is what the
pipeline depends on, so it can later be swapped for Chroma + BGE-M3 embeddings
without touching the rest of the code.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[1] / "knowledge"
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


@lru_cache(maxsize=1)
def load_entries() -> tuple[dict, ...]:
    entries = []
    for fname in ("cbc_seed.json", "ingested.json"):
        path = KB_DIR / fname
        if path.exists():
            data = json.loads(path.read_text())
            for e in data.get("entries", []):
                e.setdefault("status", "seed" if fname == "cbc_seed.json" else "ingested")
                e.setdefault("direction", None)
                e.setdefault("source_url", None)
                entries.append(e)
    return tuple(entries)


def clear_cache():
    load_entries.cache_clear()


def _bm25(query: list[str], docs: list[list[str]], k1=1.5, b=0.75) -> list[float]:
    if not docs:
        return []
    avg = sum(len(d) for d in docs) / len(docs) or 1
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for q in query:
            if q not in tf:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            s += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return scores


def retrieve(test_key: str | None, direction: str | None, test_name: str = "", k: int = 2) -> list[dict]:
    """Best notes for one abnormal result."""
    entries = list(load_entries())
    chosen: list[dict] = []
    # 1) exact key + direction, then key with no direction
    for e in entries:
        if e["test_key"] == test_key and e["direction"] == direction and direction:
            chosen.append(e)
    for e in entries:
        if e["test_key"] == test_key and e["direction"] is None and e not in chosen:
            chosen.append(e)
    # 2) ingested text matched by keywords
    ingested = [e for e in entries if e.get("status") == "ingested" and e not in chosen]
    if ingested and len(chosen) < k:
        q = _tokens(f"{test_name} {direction or ''}")
        scores = _bm25(q, [_tokens(e["title"] + " " + e["text"]) for e in ingested])
        ranked = sorted(zip(scores, ingested), key=lambda x: -x[0])
        chosen += [e for s, e in ranked if s > 0][: k - len(chosen)]
    return [{k_: e[k_] for k_ in ("id", "title", "text", "source_url", "status")} for e in chosen[:k]]


def general_note() -> dict | None:
    for e in load_entries():
        if e["test_key"] == "general":
            return {k_: e[k_] for k_ in ("id", "title", "text", "source_url", "status")}
    return None
