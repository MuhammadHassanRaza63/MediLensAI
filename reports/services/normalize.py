"""Test-name, unit and range normalisation (CBC)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process

# canonical key -> (display name, synonyms)
TESTS: dict[str, tuple[str, list[str]]] = {
    "hemoglobin": ("Hemoglobin", ["hemoglobin", "haemoglobin", "hb", "hgb", "hemoglobin hb", "haemoglobin hb", "hb hemoglobin"]),
    "rbc": ("RBC count", ["rbc", "rbc count", "red blood cell", "red blood cells", "red blood cell count", "red cell count", "erythrocytes", "total rbc count", "total rbc"]),
    "hematocrit": ("Hematocrit (PCV)", ["hematocrit", "haematocrit", "hct", "pcv", "packed cell volume", "hematocrit pcv", "haematocrit pcv", "pcv hct"]),
    "mcv": ("MCV", ["mcv", "mean corpuscular volume", "mean cell volume"]),
    "mch": ("MCH", ["mch", "mean corpuscular hemoglobin", "mean cell hemoglobin", "mean corpuscular haemoglobin"]),
    "mchc": ("MCHC", ["mchc", "mean corpuscular hemoglobin concentration", "mean corpuscular haemoglobin concentration"]),
    "rdw": ("RDW", ["rdw", "rdw cv", "rdwcv", "red cell distribution width", "rdw-cv"]),
    "wbc": ("WBC count", ["wbc", "wbc count", "white blood cell", "white blood cells", "white blood cell count", "white cell count", "tlc", "total leucocyte count", "total leukocyte count", "total wbc count", "leukocytes", "leucocytes"]),
    "platelets": ("Platelets", ["platelet", "platelets", "platelet count", "plt", "thrombocytes", "platelets count"]),
    "mpv": ("MPV", ["mpv", "mean platelet volume"]),
    "neutrophils_pct": ("Neutrophils %", ["neutrophils", "neutrophil", "neutrophils %", "neutrophils percent", "neut"]),
    "lymphocytes_pct": ("Lymphocytes %", ["lymphocytes", "lymphocyte", "lymphocytes %", "lymph"]),
    "monocytes_pct": ("Monocytes %", ["monocytes", "monocyte", "monocytes %", "mono"]),
    "eosinophils_pct": ("Eosinophils %", ["eosinophils", "eosinophil", "eosinophils %", "eos"]),
    "basophils_pct": ("Basophils %", ["basophils", "basophil", "basophils %", "baso"]),
}

# LOINC codes. hemoglobin/hematocrit/platelets/mpv were confirmed on loinc.org during
# research; the rest are from general knowledge and should be checked before export.
LOINC = {
    "hemoglobin": ("718-7", True), "hematocrit": ("4544-3", True), "platelets": ("777-3", True),
    "mpv": ("32623-1", True), "wbc": ("6690-2", False), "rbc": ("789-8", False),
    "mcv": ("787-2", False), "mch": ("785-6", False), "mchc": ("786-4", False), "rdw": ("788-0", False),
}

_SYN_INDEX: dict[str, str] = {}
for _key, (_disp, _syns) in TESTS.items():
    for _s in _syns:
        _SYN_INDEX[_s] = _key


def _clean(name: str) -> str:
    n = name.lower().replace("-", " ")
    n = re.sub(r"[^a-z0-9% ]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def canonical_key(name: str) -> Optional[str]:
    """Map a printed test name to a canonical key, or None if unknown."""
    candidates = [name, name.replace(".", "")]
    no_paren = re.sub(r"\(.*?\)", " ", name)
    candidates.append(no_paren)
    candidates += re.findall(r"\((.*?)\)", name)
    for cand in candidates:
        c = _clean(cand)
        if not c:
            continue
        if c in _SYN_INDEX:
            return _SYN_INDEX[c]
    for cand in candidates:
        c = _clean(cand)
        if len(c) >= 6:
            hit = process.extractOne(c, list(_SYN_INDEX), scorer=fuzz.ratio, score_cutoff=88)
            if hit:
                return _SYN_INDEX[hit[0]]
    return None


def display_name(key: Optional[str], fallback: str) -> str:
    return TESTS[key][0] if key in TESTS else fallback


_RANGE_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)", re.I)


def parse_range(text: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    m = _RANGE_RX.search(text.replace(",", ""))
    if not m:
        return None, None
    lo, hi = float(m.group(1)), float(m.group(2))
    return (lo, hi) if lo <= hi else (None, None)


@dataclass
class NormalizedTest:
    key: Optional[str]
    name: str                      # display name
    printed_name: str
    value: float                   # converted to canonical unit
    unit: Optional[str]            # canonical unit (or printed unit if unknown test)
    ref_low: Optional[float]
    ref_high: Optional[float]
    ref_text: Optional[str]        # as printed
    loinc: Optional[str] = None
    loinc_verified: Optional[bool] = None
    confidence: float = 1.0
    flag_raw: Optional[str] = None
    notes: list[str] = field(default_factory=list)


_CANON_UNIT = {
    "hemoglobin": "g/dL", "mchc": "g/dL", "hematocrit": "%", "mcv": "fL", "mch": "pg", "rdw": "%", "mpv": "fL",
    "wbc": "10^3/uL", "platelets": "10^3/uL", "rbc": "10^6/uL",
}


def _unit_factor(key: str, value: float, unit: Optional[str], ref_hi: Optional[float] = None) -> tuple[float, Optional[str]]:
    """Return (factor, note). value*factor is in the canonical unit.

    OCR often garbles units ("x10^3/uL" -> "x10*3/uL" or just "/uL"), and a wrong scale turns a normal
    count into a false critical alarm. So the scale is decided, in order, by:
      1. the printed reference range (printed in the same unit as the value, and numeric, so OCR rarely breaks it);
      2. an unambiguous unit text;
      3. the magnitude of the value.
    """
    u = (unit or "").lower().replace(" ", "").replace("\u00b5", "u").replace("\u03bc", "u")
    thousands = any(t in u for t in ("10^3", "10^9", "10e3", "10e9", "x10", "k/", "thou"))
    per_micro = any(t in u for t in ("/cumm", "/cmm", "/mm3", "cells/ul", "/ul"))

    if key in ("hemoglobin", "mchc"):
        limit = 30 if key == "hemoglobin" else 100       # g/dL values: Hb ~12-17, MCHC ~32-36
        if ref_hi is not None:
            return (0.1, "converted g/L to g/dL (scale taken from the printed range)") if ref_hi > limit else (1.0, None)
        if u == "g/l" or value > limit:
            return 0.1, "converted g/L to g/dL"
        return 1.0, None

    if key in ("wbc", "platelets"):
        range_per_micro = 1000 if key == "wbc" else 5000   # printed range top: ~11 vs ~11000, ~400 vs ~400000
        raw_count_min = 500 if key == "wbc" else 2000      # a value this big can only be a per-uL count
        if ref_hi is not None:
            factor = 0.001 if ref_hi >= range_per_micro else 1.0
            note = "converted /uL to 10^3/uL" if factor != 1.0 else None
            if (factor == 1.0 and per_micro and not thousands) or (factor != 1.0 and thousands):
                note = "unit text looks unclear; scale taken from the printed range"
            return factor, note
        if thousands:
            return 1.0, None
        if per_micro and value >= raw_count_min:
            return 0.001, "converted /uL to 10^3/uL"
        if per_micro:
            return 1.0, "unit text looks unclear; assumed 10^3/uL because the value is small"
        if value >= raw_count_min:
            return 0.001, "unit missing; assumed cells per uL (converted to 10^3/uL)"
        return 1.0, "unit missing; assumed 10^3/uL"

    if key == "rbc":
        if ref_hi is not None:
            return (1e-6, "converted /uL to 10^6/uL") if ref_hi >= 100000 else (1.0, None)
        if value >= 100000:
            return 1e-6, "converted /uL to 10^6/uL"
        return 1.0, None
    return 1.0, None


def normalize_test(printed_name: str, value: float, unit: Optional[str], ref_range: Optional[str],
                   confidence: float = 1.0, flag_raw: Optional[str] = None) -> NormalizedTest:
    key = canonical_key(printed_name)
    lo, hi = parse_range(ref_range)
    notes: list[str] = []
    if key is None:
        return NormalizedTest(None, printed_name.strip(), printed_name.strip(), value, unit, lo, hi,
                              ref_range, None, None, confidence, flag_raw,
                              ["Test not in the CBC list; only the report's own range can be used."])
    factor, note = _unit_factor(key, value, unit, hi)
    if note:
        notes.append(note)
    # A printed range is in the same unit as the printed value, so scale it with the value.
    if lo is not None and hi is not None:
        lo, hi = lo * factor, hi * factor
    code, verified = LOINC.get(key, (None, None))
    return NormalizedTest(
        key=key, name=display_name(key, printed_name), printed_name=printed_name.strip(),
        value=round(value * factor, 4), unit=_CANON_UNIT.get(key, unit),
        ref_low=None if lo is None else round(lo, 4), ref_high=None if hi is None else round(hi, 4),
        ref_text=ref_range, loinc=code, loinc_verified=verified,
        confidence=confidence, flag_raw=flag_raw, notes=notes,
    )
