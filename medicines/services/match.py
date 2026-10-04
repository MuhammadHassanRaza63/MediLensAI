"""Match what was read from the pack against a drug table (DRAP-style columns)."""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from rapidfuzz import fuzz

from .extract import MedicineFields, parse_form, parse_strengths

DEFAULT_TABLE = Path(__file__).resolve().parents[1] / "data" / "drugs_sample.csv"

MIN_CANDIDATE_SCORE = 60      # below this a fuzzy hit is treated as noise

ALIASES = {"acetaminophen": "paracetamol", "albuterol": "salbutamol", "amoxycillin": "amoxicillin",
           "acetylsalicylic acid": "aspirin", "asa": "aspirin"}


@lru_cache(maxsize=2)
def _load(path: str) -> tuple[dict, ...]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["_ingredients"] = [p.strip().lower() for p in r["ingredient"].split("+")]
        r["_strengths"] = [s.strip().lower() for s in r["strength"].split("+")]
    return tuple(rows)


def load_drugs() -> tuple[dict, ...]:
    return _load(str(getattr(settings, "MEDILENS_DRUG_TABLE", DEFAULT_TABLE)))


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z][a-z\-]+", (text or "").lower())
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


def _norm_ing(s: str) -> str:
    s = s.lower().strip()
    return ALIASES.get(s, s)


def _best(needle: str, haystack: list[str]) -> float:
    return max((fuzz.ratio(needle, h) for h in haystack), default=0.0)


def match_drugs(fields: MedicineFields, ocr_text: str, top: int = 3) -> list[dict]:
    text_tokens = _tokens(ocr_text)
    if fields.brand:
        text_tokens.append(fields.brand.lower())
    generic = [_norm_ing(g) for g in fields.generic_names]
    ing_pool = [_norm_ing(t) for t in text_tokens] + generic
    strengths = set(fields.strengths) | set(parse_strengths(ocr_text))
    form = (fields.form or parse_form(ocr_text) or "").lower()

    scored = []
    for row in load_drugs():
        brand = row["brand"].lower()
        brand_score = _best(brand, text_tokens) if len(brand) >= 4 else (100.0 if brand in text_tokens else 0.0)
        ing_hits = [_best(i, ing_pool) >= 85 for i in row["_ingredients"]]
        ing_frac = sum(ing_hits) / len(ing_hits)
        reasons = []
        if brand_score >= 85:
            reasons.append(f"brand text matches '{row['brand']}' ({brand_score:.0f})")
        if ing_frac == 1:
            reasons.append("ingredient(s) printed on the pack match")
        row_strengths = {s for s in row["_strengths"]}
        if strengths:
            if row_strengths <= strengths:
                s_state = "match"
                reasons.append("strength matches")
            elif row_strengths & strengths:
                s_state = "partial"
                reasons.append("only part of the strength matches")
            else:
                s_state = "conflict"
                reasons.append(f"strength on pack ({', '.join(sorted(strengths))}) differs from {row['strength']}")
        else:
            s_state = "unknown"
        score = brand_score
        if brand_score < 80 and ing_frac == 1 and s_state in ("match", "unknown"):
            score = 78.0                     # identified by ingredient only
        score += 5 if ing_frac == 1 else 0
        score += 5 if s_state == "match" else 0
        score -= 25 if s_state == "conflict" else 10 if s_state == "partial" else 0
        if form and form != row["form"].lower():
            score -= 5
            reasons.append(f"form on pack ({form}) differs from {row['form']}")
        score = max(0.0, min(100.0, score))
        if score >= MIN_CANDIDATE_SCORE:
            scored.append({"brand": row["brand"], "ingredient": row["ingredient"], "strength": row["strength"],
                           "form": row["form"], "general_use": row["general_use"], "score": round(score, 1),
                           "strength_state": s_state, "ingredient_only": brand_score < 80, "reasons": reasons})
    scored.sort(key=lambda c: -c["score"])
    return scored[:top]


def alternatives(chosen: dict) -> list[dict]:
    """Same active ingredient(s) + same strength + same dosage form, other brands."""
    out = []
    for row in load_drugs():
        if (row["ingredient"].lower() == chosen["ingredient"].lower()
                and row["strength"].lower() == chosen["strength"].lower()
                and row["form"].lower() == chosen["form"].lower()
                and row["brand"].lower() != chosen["brand"].lower()):
            out.append({"brand": row["brand"], "ingredient": row["ingredient"],
                        "strength": row["strength"], "form": row["form"]})
    return out
