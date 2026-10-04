"""Deterministic analysis: Low / Normal / High / Critical. No LLM involved."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .normalize import NormalizedTest

DATA = Path(__file__).resolve().parents[2] / "core" / "data"


@lru_cache(maxsize=1)
def load_reference_ranges() -> dict:
    return json.loads((DATA / "reference_ranges.json").read_text())


@lru_cache(maxsize=1)
def load_critical_thresholds() -> dict:
    return json.loads((DATA / "critical_thresholds.json").read_text())


@dataclass
class AnalyzedTest:
    key: Optional[str]
    name: str
    value: float
    unit: Optional[str]
    ref_low: Optional[float]
    ref_high: Optional[float]
    ref_source: str                   # report | fallback | fallback_unverified | none
    flag: str                         # Low | Normal | High | Critical | Not assessed
    direction: Optional[str] = None   # low | high (for abnormal/critical)
    critical: bool = False
    loinc: Optional[str] = None
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _fallback_range(key: str, sex: Optional[str]):
    entry = load_reference_ranges()["tests"].get(key)
    if not entry:
        return None, None, "none", []
    notes = []
    if sex in ("male", "female"):
        lo, hi = entry[sex]
    else:
        lo = min(entry["male"][0], entry["female"][0])
        hi = max(entry["male"][1], entry["female"][1])
        notes.append("Sex not given, so a wide adult range was used.")
    src = "fallback" if entry.get("verified") else "fallback_unverified"
    if src == "fallback_unverified":
        notes.append("Fallback range is not yet clinically verified.")
    return lo, hi, src, notes


def analyze_test(t: NormalizedTest, sex: Optional[str]) -> AnalyzedTest:
    lo, hi, src, notes = t.ref_low, t.ref_high, "report", list(t.notes)
    if lo is None or hi is None:
        if t.key:
            lo, hi, src, extra = _fallback_range(t.key, sex)
            notes += extra
        else:
            lo, hi, src = None, None, "none"
    if lo is None or hi is None:
        return AnalyzedTest(t.key, t.name, t.value, t.unit, None, None, "none", "Not assessed",
                            loinc=t.loinc, confidence=t.confidence,
                            notes=notes + ["No reference range available for this test."])
    if t.value < lo:
        flag, direction = "Low", "low"
    elif t.value > hi:
        flag, direction = "High", "high"
    else:
        flag, direction = "Normal", None
    if t.flag_raw:
        raw = t.flag_raw.strip().upper()[:1]
        computed = {"Low": "L", "High": "H"}.get(flag)
        if raw in ("H", "L") and raw != computed:
            notes.append(f"The report printed '{t.flag_raw}' but our check gives {flag}; please verify the values.")
    return AnalyzedTest(t.key, t.name, t.value, t.unit, lo, hi, src, flag, direction,
                        loinc=t.loinc, confidence=t.confidence, notes=notes)


def analyze_all(tests: list[NormalizedTest], sex: Optional[str]) -> list[AnalyzedTest]:
    return [analyze_test(t, sex) for t in tests]


def pattern_hints(results: list[AnalyzedTest]) -> list[str]:
    """Plain-language pattern notes. These are hints, never diagnoses."""
    by_key = {r.key: r for r in results if r.key}
    hints: list[str] = []
    hb, mcv = by_key.get("hemoglobin"), by_key.get("mcv")
    if hb and hb.flag in ("Low", "Critical") and hb.direction == "low" and mcv:
        if mcv.direction == "low":
            hints.append("Low hemoglobin together with a low MCV means the red blood cells are smaller than usual "
                         "(a microcytic pattern). This can have several causes, which a doctor can sort out.")
        elif mcv.direction == "high":
            hints.append("Low hemoglobin together with a high MCV means the red blood cells are larger than usual "
                         "(a macrocytic pattern). This can have several causes, which a doctor can sort out.")
        elif mcv.flag == "Normal":
            hints.append("Low hemoglobin with a normal MCV means the red blood cells are of usual size "
                         "(a normocytic pattern). This can have several causes, which a doctor can sort out.")
    plt, wbc = by_key.get("platelets"), by_key.get("wbc")
    if plt and plt.direction == "low" and wbc and wbc.direction == "low" and hb and hb.direction == "low":
        hints.append("Hemoglobin, white cells and platelets are all below range. Together this deserves "
                     "prompt attention from a doctor.")
    return hints


def apply_triage(results: list[AnalyzedTest]) -> list[dict]:
    """Mark critical values using the (sign-off pending) threshold file."""
    cfg = load_critical_thresholds()
    alerts = []
    for r in results:
        limits = cfg["tests"].get(r.key or "")
        if not limits or r.flag == "Not assessed":
            continue
        if r.value < limits["low"]:
            r.critical, r.flag, r.direction = True, "Critical", "low"
        elif r.value > limits["high"]:
            r.critical, r.flag, r.direction = True, "Critical", "high"
        else:
            continue
        alerts.append({"test": r.name, "value": r.value, "unit": r.unit, "direction": r.direction,
                       "threshold_status": cfg.get("status")})
    return alerts
