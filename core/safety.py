"""Hard output rules: no diagnosis, no dosage, no start/stop-medicine advice.

``check_policy`` is deliberately conservative. A false positive only costs us a
regeneration or the safe template; a false negative could harm a user.
"""
from __future__ import annotations

import re

DISCLAIMERS = {
    "en": ("This is general information to help you understand your report. "
           "It is not a diagnosis and not medical advice. Please discuss your results "
           "with a doctor."),
    "roman_urdu": ("Ye sirf aam maloomat hai jo aap ki report samajhne mein madad karti hai. "
                   "Ye tashkhees (diagnosis) ya medical mashwara nahi hai. "
                   "Apne nataij ke baare mein doctor se zaroor baat karein."),
}

MEDICINE_DISCLAIMERS = {
    "en": "Please confirm this with your doctor or pharmacist before using any medicine.",
    "roman_urdu": "Koi bhi dawa istemal karne se pehle apne doctor ya pharmacist se confirm zaroor karein.",
}

CRITICAL_WARNINGS = {
    "en": ("One or more of your results is in a range that can be urgent. "
           "Please contact a doctor or go to an emergency department as soon as possible."),
    "roman_urdu": ("Aap ke kuch nataij aise range mein hain jo urgent ho sakte hain. "
                   "Meherbani kar ke fori taur par doctor se rabta karein ya emergency mein jayein."),
}

# (regex, label). Case-insensitive.
_RULES = [
    (r"\byou (have|are suffering from|are diagnosed with|definitely have|probably have)\b", "diagnosis"),
    (r"\byour diagnosis is\b", "diagnosis"),
    (r"\bthis (confirms|proves) (that )?you\b", "diagnosis"),
    (r"\baap\s+ko\s+[\w\s-]{0,30}?\bhai\b", "diagnosis"),
    (r"\baap\s+(is\s+)?(bimari|marz)\s+mein\b", "diagnosis"),
    (r"\b\d+(\.\d+)?\s*(mg|mcg|µg|ug|ml|iu|g)\b[^.\n]{0,40}\b(daily|twice|thrice|once|every|per day|a day|din mein|roz|baar)\b", "dosage"),
    (r"\b(once|twice|thrice)\s+(a|per)\s+day\b", "dosage"),
    (r"\b(dose|dosage|dosing)\s+(is|of|should|will)\b", "dosage"),
    (r"\b(take|use|swallow|consume)\s+(this|the|these|it)\b[^.\n]{0,40}\b(medicine|medication|tablet|tablets|capsule|syrup|drug|pill)s?\b", "start_medicine"),
    (r"\byou should (take|start|use)\b", "start_medicine"),
    (r"\b(ye|yeh|is)\s+(dawa|dawai|tablet|goli|syrup)\s+(lein|len|khayein|khaye|istemal karein|shuru karein)\b", "start_medicine"),
    (r"\b(dawa|dawai|tablet|goli)\s+(lein|len|khayein|khaye|piyein)\b", "start_medicine"),
    (r"\b(stop|discontinue|quit|skip|double|increase|decrease|reduce)\s+(taking|your|the)\b[^.\n]{0,30}\b(medicine|medication|dose|tablet|tablets|pills?)\b", "stop_medicine"),
    (r"\b(dawa|dawai|tablet|goli)\s+(chhor|chor|band)\s+(dein|den|kar dein|kar den)\b", "stop_medicine"),
]
_COMPILED = [(re.compile(p, re.I), label) for p, label in _RULES]


def check_policy(text: str) -> list[dict]:
    """Return a list of violations: [{"type": ..., "match": ...}]."""
    found = []
    for rx, label in _COMPILED:
        m = rx.search(text or "")
        if m:
            found.append({"type": label, "match": m.group(0)})
    return found


def is_safe(text: str) -> bool:
    return not check_policy(text)
