"""Read printed text/fields from a medicine pack photo."""
from __future__ import annotations

import io
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from core.llm import extract_json, get_llm

log = logging.getLogger(__name__)

STRENGTH_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|g|ml|iu)\b", re.I)
FORMS = {"tablet": "tablet", "tablets": "tablet", "tab": "tablet", "capsule": "capsule", "capsules": "capsule",
         "cap": "capsule", "syrup": "syrup", "suspension": "suspension", "injection": "injection",
         "cream": "cream", "ointment": "ointment", "drops": "drops"}


class MedicineFields(BaseModel):
    brand: Optional[str] = None
    generic_names: list[str] = []
    strengths: list[str] = []          # e.g. ["500 mg"]
    form: Optional[str] = None
    manufacturer: Optional[str] = None
    expiry: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def norm_strength(num: str, unit: str) -> str:
    unit = unit.lower().replace("µg", "mcg").replace("ug", "mcg")
    n = float(num)
    return f"{int(n) if n == int(n) else n} {unit}"


def parse_strengths(text: str) -> list[str]:
    seen, out = set(), []
    for num, unit in STRENGTH_RX.findall(text or ""):
        s = norm_strength(num, unit)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def parse_form(text: str) -> Optional[str]:
    for tok in re.findall(r"[a-z]+", (text or "").lower()):
        if tok in FORMS:
            return FORMS[tok]
    return None


def ocr_text_from_image(data: bytes) -> str:
    import pytesseract
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    g = ImageOps.autocontrast(ImageOps.grayscale(img))
    if g.width < 1600:
        s = 1600 / g.width
        g = g.resize((int(g.width * s), int(g.height * s)))
    # Packs have scattered text blocks, so use sparse-text mode and keep both orientations of reading simple.
    return pytesseract.image_to_string(g, config="--psm 11")


VISION_SYSTEM = (
    "You read the printed text on a medicine package photo. Return ONLY JSON: "
    '{"brand":str|null,"generic_names":[str],"strengths":["500 mg"],"form":"tablet|capsule|syrup|...|null",'
    '"manufacturer":str|null,"expiry":str|null,"confidence":0-1}. '
    "Copy text exactly as printed. If something is not visible, use null or an empty list. Never guess a "
    "brand or ingredient that is not printed. Lower the confidence if text is cut off, blurry or reflective.")


def vision_fields(data: bytes, media_type: str) -> MedicineFields:
    raw = get_llm().vision(VISION_SYSTEM, "Read this medicine pack.", data, media_type)
    return MedicineFields(**extract_json(raw))


def fields_from_text(text: str) -> MedicineFields:
    return MedicineFields(strengths=parse_strengths(text), form=parse_form(text), confidence=0.0)


def extract_fields(data: Optional[bytes], media_type: str = "image/jpeg",
                   ocr_text: Optional[str] = None) -> tuple[MedicineFields, str, list[str]]:
    """Return (fields, ocr_text, notes). ocr_text may be supplied by the client (and is used as-is)."""
    notes: list[str] = []
    text = ocr_text or ""
    if not text and data is not None:
        try:
            text = ocr_text_from_image(data)
        except Exception as exc:
            notes.append(f"OCR failed: {exc}")
    fields = fields_from_text(text)
    llm = get_llm()
    if llm.available and data is not None:
        for attempt in range(2):
            try:
                lf = vision_fields(data, media_type)
                fields = MedicineFields(
                    brand=lf.brand, generic_names=lf.generic_names,
                    strengths=sorted(set(lf.strengths) | set(fields.strengths)) if lf.strengths else fields.strengths,
                    form=lf.form or fields.form, manufacturer=lf.manufacturer, expiry=lf.expiry,
                    confidence=lf.confidence)
                break
            except Exception as exc:   # bad JSON, schema failure, or a network/API error
                log.warning("medicine vision attempt %s failed: %s", attempt + 1, exc)
        else:
            notes.append("The vision model could not be used; using OCR text only.")
    return fields, text, notes
