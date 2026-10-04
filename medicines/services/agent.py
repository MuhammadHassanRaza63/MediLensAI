"""Medicine identification agent: quality -> extract -> match -> confidence gate -> guarded output."""
from __future__ import annotations

from typing import Optional

from django.conf import settings

from core.safety import MEDICINE_DISCLAIMERS, check_policy

from .extract import extract_fields
from .match import alternatives, load_drugs, match_drugs
from .quality import assess_image


def _guard(text: str) -> Optional[str]:
    """Never pass through dosage / start-stop language."""
    return None if check_policy(text) else text


def _result_card(c: dict, language: str) -> dict:
    return {
        "brand": c["brand"], "active_ingredient": c["ingredient"], "strength": c["strength"], "form": c["form"],
        "general_use": _guard(c.get("general_use", "")),
        "alternatives": alternatives(c),
        "alternatives_note": ("Same active ingredient, strength and dosage form only. This is information, "
                              "not a recommendation to switch. Ask your pharmacist."),
        "disclaimer": MEDICINE_DISCLAIMERS.get(language, MEDICINE_DISCLAIMERS["en"]),
    }


def identify(data: Optional[bytes], media_type: str = "image/jpeg", ocr_text: Optional[str] = None,
             language: str = "en") -> dict:
    language = language if language in MEDICINE_DISCLAIMERS else "en"
    disclaimer = MEDICINE_DISCLAIMERS[language]
    quality = assess_image(data) if data is not None else {"ok": True, "issues": [], "note": "no image; text only"}
    fields, text, notes = extract_fields(data, media_type, ocr_text)
    notes = notes + quality.get("warnings", [])
    base = {"quality": quality, "extracted": fields.model_dump(), "ocr_text": text[:2000],
            "notes": notes, "disclaimer": disclaimer}

    candidates = match_drugs(fields, text)
    if not quality["ok"] and (not candidates or candidates[0]["score"] < settings.MEDILENS_MED_HIGH_CONFIDENCE):
        return {**base, "status": "retake_photo", "candidates": [],
                "message": "Please take a clearer photo of the front of the pack. " + " ".join(quality["issues"])}
    if not candidates or candidates[0]["score"] < settings.MEDILENS_MED_MEDIUM_CONFIDENCE:
        return {**base, "status": "retake_photo", "candidates": candidates,
                "message": "The medicine could not be identified. Please retake the photo so the brand name "
                           "and strength are clearly visible, or type the name."}

    best = candidates[0]
    ambiguous = (len(candidates) > 1 and best["score"] - candidates[1]["score"] < 5
                 and candidates[1]["ingredient"] != best["ingredient"])
    sure = (best["score"] >= settings.MEDILENS_MED_HIGH_CONFIDENCE and best["strength_state"] != "conflict"
            and not best["ingredient_only"] and not ambiguous and quality["ok"])
    if sure:
        return {**base, "status": "identified", "candidates": candidates[:1], "result": _result_card(best, language)}
    return {**base, "status": "needs_confirmation", "candidates": candidates,
            "message": ("Is this one of these medicines? Please confirm, or retake the photo. "
                        "If the strength or ingredient on your pack differs, tell us."),
            "confirm_hint": "POST /api/medicines/confirm/ with {brand, strength, form}."}


def confirm(brand: str, strength: Optional[str] = None, form: Optional[str] = None, language: str = "en") -> Optional[dict]:
    brand_l = (brand or "").strip().lower()
    rows = [r for r in load_drugs() if r["brand"].lower() == brand_l
            and (not strength or r["strength"].lower() == strength.strip().lower())
            and (not form or r["form"].lower() == form.strip().lower())]
    if not rows:
        return None
    r = rows[0]
    card = _result_card({"brand": r["brand"], "ingredient": r["ingredient"], "strength": r["strength"],
                         "form": r["form"], "general_use": r["general_use"]}, language)
    return {"status": "confirmed_by_user", "result": card,
            "ambiguous_strengths": sorted({x["strength"] for x in rows}) if len(rows) > 1 and not strength else []}
