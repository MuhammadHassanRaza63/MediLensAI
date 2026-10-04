"""Verification layer: the explanation must be consistent with the validated data."""
from __future__ import annotations

import json
import re

from core.llm import BaseLLM, extract_json
from core.safety import check_policy

from .explain import counts, fmt, short_name

_UNIT_POW = re.compile(r"10\s*\^\s*\d+|x\s*10\s*\d+", re.I)
_LIST_MARK = re.compile(r"(?m)^\s*\d+[.)]\s+")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    t = _UNIT_POW.sub(" ", _LIST_MARK.sub("", text))
    return [float(x) for x in _NUM.findall(t)]


def verify_explanation(text: str, results: list[dict], hints: list[str], knowledge: dict,
                       llm: BaseLLM | None = None) -> dict:
    checks = []

    # 1) policy: no diagnosis / dosage / start-stop medicine language
    violations = check_policy(text)
    checks.append({"name": "policy", "passed": not violations, "detail": violations})

    # 2) every number in the text must come from the data (or the retrieved notes)
    allowed: set[float] = set()
    for r in results:
        for k in ("value", "ref_low", "ref_high"):
            if r.get(k) is not None:
                allowed.add(round(float(r[k]), 4))
    c = counts(results)
    allowed |= {float(v) for v in c.values()}
    for src in [n["text"] for notes in knowledge.values() for n in notes] + hints:
        allowed |= set(_numbers(src))
    bad = sorted({n for n in _numbers(text) if not any(abs(n - a) < 1e-4 for a in allowed)})
    checks.append({"name": "numbers_match_data", "passed": not bad, "detail": bad})

    # 3) every abnormal result is mentioned with its value
    missing = []
    low = text.lower()
    for r in results:
        if r["flag"] in ("Low", "High", "Critical"):
            if short_name(r["name"]) not in low or fmt(r["value"]) not in text:
                missing.append(r["name"])
    checks.append({"name": "abnormal_results_covered", "passed": not missing, "detail": missing})

    # 4) optional LLM judge for unsupported claims (best-effort, only when a model is configured)
    if llm is not None and llm.available:
        try:
            facts = json.dumps({"results": results, "notes": [n["text"] for v in knowledge.values() for n in v],
                                "pattern_notes": hints}, ensure_ascii=False)
            raw = llm.complete(
                "You are a strict fact-checker. Given FACTS and a TEXT, list every sentence in TEXT that makes a "
                "medical or numeric claim not supported by FACTS. Return JSON only: {\"unsupported\": [\"sentence\", ...]}.",
                f"FACTS:\n{facts}\n\nTEXT:\n{text}", max_tokens=600)
            unsupported = extract_json(raw).get("unsupported", [])
            checks.append({"name": "claims_supported_by_sources", "passed": not unsupported, "detail": unsupported})
        except Exception as exc:  # judge failure must not pass silently
            checks.append({"name": "claims_supported_by_sources", "passed": False,
                           "detail": [f"judge unavailable: {exc}"]})
    else:
        checks.append({"name": "claims_supported_by_sources", "passed": True, "detail": "skipped (no LLM configured)"})

    return {"passed": all(c_["passed"] for c_ in checks), "checks": checks}
