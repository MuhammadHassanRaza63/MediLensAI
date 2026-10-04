"""Explanation writer. The LLM (when available) only rewrites validated facts;
a deterministic template is always available as the safe fallback."""
from __future__ import annotations

import json
import re

from core.llm import BaseLLM, LLMUnavailable


def fmt(v) -> str:
    if v is None:
        return ""
    s = ("%f" % round(float(v), 4)).rstrip("0").rstrip(".")
    return s or "0"


def short_name(name: str) -> str:
    return re.split(r"[ (%]", name)[0].lower()


def counts(results) -> dict:
    assessed = [r for r in results if r["flag"] != "Not assessed"]
    abnormal = [r for r in assessed if r["flag"] != "Normal"]
    return {"total": len(results), "normal": len(assessed) - len(abnormal),
            "abnormal": len(abnormal), "not_assessed": len(results) - len(assessed),
            "critical": sum(1 for r in results if r.get("critical"))}


def _range_label(r, lang):
    src = "range" if r["ref_source"] == "report" else ("standard adult range" if lang == "en" else "standard adult range")
    return f"{src}: {fmt(r['ref_low'])} - {fmt(r['ref_high'])}"


def template_explanation(results: list[dict], hints: list[str], knowledge: dict, language: str = "en") -> str:
    """knowledge: {test_key: [note,...], "_general": [note]}."""
    c = counts(results)
    abnormal = [r for r in results if r["flag"] not in ("Normal", "Not assessed")]
    normal = [r for r in results if r["flag"] == "Normal"]
    unassessed = [r for r in results if r["flag"] == "Not assessed"]
    ur = language == "roman_urdu"
    lines: list[str] = []

    if ur:
        lines.append(f"Jaiza: Aap ki report mein {c['total']} nataij parhe ja sake: "
                     f"{c['normal']} range ke andar aur {c['abnormal']} range se bahar.")
    else:
        lines.append(f"Overview: {c['total']} results could be read from your report: "
                     f"{c['normal']} inside the range and {c['abnormal']} outside it.")
    if unassessed:
        names = ", ".join(r["name"] for r in unassessed)
        lines.append((f"In ke liye range maujood nahi thi, is liye check nahi hue: {names}." if ur
                      else f"No reference range was available for: {names}, so they were not assessed."))

    if abnormal:
        lines.append("")
        lines.append("Range se bahar nataij:" if ur else "Results outside the range:")
        for r in abnormal:
            unit = f" {r['unit']}" if r.get("unit") else ""
            if ur:
                where = "range se kam hai" if r["direction"] == "low" else "range se zyada hai"
            else:
                where = "lower than the range" if r["direction"] == "low" else "higher than the range"
            lines.append(f"- {r['name']}: {fmt(r['value'])}{unit} ({_range_label(r, language)}) - {where}.")
            for note in knowledge.get(r["key"] or "", [])[:2]:
                lines.append(f"  {note['text']}")
    if hints:
        lines.append("")
        lines.append("Pattern notes (English):" if ur else "Pattern notes:")
        lines += [f"- {h}" for h in hints]
    if normal:
        lines.append("")
        lines.append(("Range ke andar nataij: " if ur else "Inside the range: ") + ", ".join(r["name"] for r in normal) + ".")
    lines.append("")
    lines.append("Agla qadam: Meherbani kar ke ye report doctor ko dikhayein. Wo aap ki alamaat aur history ke saath "
                 "isay parh sakte hain." if ur else
                 "Next step: Please show this report to a doctor, who can read it together with your symptoms and history.")
    return "\n".join(lines)


LLM_SYSTEM = (
    "You explain lab results to a non-expert patient. Rules: use ONLY the facts in the JSON you are given; "
    "never add, compute or round a number; never state or imply a diagnosis (do not write 'you have ...'); "
    "never mention medicines, doses, or what to take or stop; keep medical test names in English; "
    "be calm and plain. Structure: a one-sentence overview with the counts, one bullet per result outside the "
    "range (value, unit, range, simple meaning taken from 'knowledge'), the pattern notes if any, a short list of "
    "the normal tests, and a closing line recommending the person show the report to a doctor. "
    "Do NOT write a disclaimer or an emergency warning; they are added separately.")


def llm_explanation(llm: BaseLLM, results, hints, knowledge, language, feedback: str | None = None) -> str:
    lang_instr = ("Write in Roman Urdu (Urdu written in English letters)." if language == "roman_urdu"
                  else "Write in simple English.")
    payload = {
        "counts": counts(results),
        "results": [{k: r[k] for k in ("name", "value", "unit", "ref_low", "ref_high", "ref_source", "flag", "direction")}
                    for r in results],
        "pattern_notes": hints,
        "knowledge": {k: [n["text"] for n in v] for k, v in knowledge.items() if k != "_general"},
    }
    user = f"{lang_instr}\n\nDATA:\n{json.dumps(payload, ensure_ascii=False)}"
    if feedback:
        user += f"\n\nYour previous draft was rejected for: {feedback}. Fix this and follow all rules."
    return llm.complete(LLM_SYSTEM, user, max_tokens=1400).strip()
