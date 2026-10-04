"""Orchestrates: normalize -> analyze (rules) -> triage -> retrieve -> explain -> verify."""
from __future__ import annotations

from typing import Optional

from django.conf import settings

from core.llm import get_llm
from core.safety import CRITICAL_WARNINGS, DISCLAIMERS, check_policy

from . import rag
from .analyze import analyze_all, apply_triage, load_critical_thresholds, pattern_hints
from .explain import llm_explanation, template_explanation
from .normalize import normalize_test
from .schemas import ExtractionResult
from .verify import verify_explanation


def extraction_needs_confirmation(ex: ExtractionResult) -> list[str]:
    reasons = []
    if not ex.tests:
        reasons.append("No test results could be read from this file.")
    low = [t.test for t in ex.tests if t.confidence < settings.MEDILENS_MIN_EXTRACTION_CONFIDENCE]
    if low:
        reasons.append("Please confirm these values (low reading confidence): " + ", ".join(low))
    reasons += [w for w in ex.warnings if "confirm" in w.lower() or "check the values" in w.lower()]
    return reasons


def _write_explanation(results, hints, knowledge, language, llm):
    """Return (text, source, verification). Tries LLM (twice), falls back to the template."""
    if llm.available:
        feedback = None
        for _ in range(2):
            try:
                text = llm_explanation(llm, results, hints, knowledge, language, feedback)
            except Exception:
                break
            ver = verify_explanation(text, results, hints, knowledge, llm)
            if ver["passed"]:
                return text, "llm", ver
            feedback = "; ".join(f"{c['name']}: {c['detail']}" for c in ver["checks"] if not c["passed"])
    text = template_explanation(results, hints, knowledge, language)
    ver = verify_explanation(text, results, hints, knowledge, None)
    return text, ("template_fallback" if llm.available else "template"), ver


def run_analysis(extraction: ExtractionResult, language: str = "en",
                 sex: Optional[str] = None, age: Optional[int] = None) -> dict:
    language = language if language in DISCLAIMERS else "en"
    sex = sex or extraction.patient.sex
    age = age or extraction.patient.age

    normalized = [normalize_test(t.test, t.value, t.unit, t.ref_range, t.confidence, t.flag_raw)
                  for t in extraction.tests]
    analyzed = analyze_all(normalized, sex)
    alerts = apply_triage(analyzed)
    hints = pattern_hints(analyzed)
    results = [r.to_dict() for r in analyzed]

    knowledge: dict = {}
    citations: dict = {}
    for r in results:
        if r["flag"] in ("Low", "High", "Critical") and r["key"]:
            notes = rag.retrieve(r["key"], r["direction"], r["name"])
            if notes:
                knowledge[r["key"]] = notes
                for n in notes:
                    citations[n["id"]] = {k: n[k] for k in ("id", "title", "source_url", "status")}

    llm = get_llm()
    body, source, verification = _write_explanation(results, hints, knowledge, language, llm)
    if check_policy(body):  # last line of defence
        body = template_explanation(results, hints, knowledge, language)
        source = "template_fallback"
        verification = verify_explanation(body, results, hints, knowledge, None)

    urgent = CRITICAL_WARNINGS[language] if alerts else None
    parts = ([urgent] if urgent else []) + [body, DISCLAIMERS[language]]
    seed_used = any(c["status"] == "seed" for c in citations.values())
    return {
        "status": "completed",
        "language": language,
        "patient": {"sex": sex, "age": age},
        "results": results,
        "hints": hints,
        "alerts": alerts,
        "urgent_warning": urgent,
        "explanation": "\n\n".join(parts),
        "explanation_source": source,
        "citations": list(citations.values()),
        "verification": verification,
        "notices": (
            (["Critical-value limits are placeholders awaiting clinical sign-off."]
             if load_critical_thresholds().get("signed_off_by") is None else [])
            + (["Background notes are developer seed text, not yet clinically reviewed."] if seed_used else [])
        ),
    }
