"""Turn an uploaded file (or pasted text) into a validated ExtractionResult.

Paths:
  * pasted text / digital PDF text  -> regex line parser          (high confidence)
  * scanned PDF / photo             -> Tesseract OCR + regex parser (capped confidence, user confirms)
  * scanned PDF / photo + LLM       -> vision LLM JSON, cross-checked against OCR
"""
from __future__ import annotations

import io
import logging
import re
from typing import Optional

from pydantic import ValidationError

from core.llm import extract_json, get_llm

from .normalize import canonical_key
from .schemas import ExtractedTest, ExtractionResult, Patient

log = logging.getLogger(__name__)

OCR_CONFIDENCE_CAP = 0.75   # OCR alone is never trusted without user confirmation

_NAME_VALUE_RX = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z .,()/%#\-]*?)[\s:.\-]*(?P<rest>[<>]?\s*\d.*)$")
_VALUE_RX = re.compile(r"^[<>]?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
_RANGE_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)", re.I)
_FLAG_RX = re.compile(r"(?<!\S)\(?(HH|LL|H|L|High|Low)\)?\*?(?!\S)")
_JUNK_NAME_RX = re.compile(r"\b(date|page|phone|tel|mobile|age|ref|reference|id|sample|report|collected|received|reg|mr|lab|time|dr|address|cnic)\b", re.I)
_UNIT_RX = re.compile(
    r"(?:x\s*)?10\s*[\^*°%'`]*\s*\d+\s*/\s*(?:[µμu])?[lL]|g/d[lL]|g/[lL]|mg/d[lL]|\bf[lL]\b|\bpg\b|%|"
    r"(?:mill(?:ion)?|m|k|thou(?:sand)?)\s*/\s*(?:cumm|cmm|mm3|[µμu][lL])|"
    r"cells\s*/\s*[µμu]?[lL]|/\s*(?:cumm|cmm|mm3|[µμu][lL])", re.I)
_SEX_RX = re.compile(r"\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f)\b", re.I)
_AGE_RX = re.compile(r"\bage\s*[:\-]?\s*(\d{1,3})\b", re.I)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_patient(text: str) -> Patient:
    sex = age = None
    m = _SEX_RX.search(text)
    if m:
        sex = "male" if m.group(1).lower() in ("m", "male") else "female"
    m = _AGE_RX.search(text)
    if m and 0 <= int(m.group(1)) <= 130:
        age = int(m.group(1))
    return Patient(sex=sex, age=age)


def parse_report_text(text: str, base_confidence: float = 0.95) -> list[ExtractedTest]:
    tests: list[ExtractedTest] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 160:
            continue
        m = _NAME_VALUE_RX.match(line)
        if not m:
            continue
        name = m.group("name").strip(" .:-,")
        rest = m.group("rest")
        vm = _VALUE_RX.match(rest.strip())
        if not vm or len(name) < 2:
            continue
        value = _num(vm.group(1))
        remainder = rest.strip()[vm.end():]
        range_m = _RANGE_RX.search(remainder)
        ref_range = None
        if range_m:
            ref_range = f"{range_m.group(1)} - {range_m.group(2)}"
            remainder = remainder[:range_m.start()] + " " + remainder[range_m.end():]
        flag_m = _FLAG_RX.search(remainder)
        flag = None
        if flag_m:
            flag = flag_m.group(1)
            remainder = remainder[:flag_m.start()] + " " + remainder[flag_m.end():]
        unit_m = _UNIT_RX.search(remainder)
        unit = re.sub(r"\s+", "", unit_m.group(0)) if unit_m else None
        if unit:
            unit = re.sub(r"10[\^*°%'`]+(\d)", r"10^\1", unit)      # OCR often mangles the superscript
        known = canonical_key(name) is not None
        if not known:
            if not ref_range or _JUNK_NAME_RX.search(name):
                continue        # junk such as "Page 1 of 2" or dates
            lo_s, hi_s = (float(x) for x in ref_range.split(" - "))
            if hi_s > 50 * max(lo_s, 1) or not (lo_s <= value <= hi_s * 5):
                continue
        conf = base_confidence if (known and ref_range) else base_confidence - 0.05 if known else base_confidence - 0.2
        try:
            tests.append(ExtractedTest(test=name, value=value, unit=unit, ref_range=ref_range,
                                       confidence=max(conf, 0.0), flag_raw=flag))
        except ValidationError:
            continue
    return tests


# --------------------------------------------------------------------- file handling
def detect_kind(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if data[:5] == b"%PDF-" or name.endswith(".pdf"):
        return "pdf"
    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")) or data[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1"):
        return "image"
    if name.endswith((".txt", ".csv")):
        return "text"
    return "unknown"


def preprocess_image(img):
    """Grayscale, autocontrast and upscale small images to help OCR."""
    from PIL import ImageOps

    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    if g.width < 1600:
        scale = 1600 / g.width
        g = g.resize((int(g.width * scale), int(g.height * scale)))
    return g


def ocr_image(img) -> str:
    import pytesseract

    return pytesseract.image_to_string(preprocess_image(img), config="--psm 6")


def _pdf_pages_text(data: bytes) -> list[str]:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return [(p.extract_text() or "") for p in pdf.pages]


def _pdf_pages_images(data: bytes):
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return [p.to_image(resolution=200).original for p in pdf.pages]


VISION_SYSTEM = (
    "You extract laboratory results from a medical report image. Return ONLY JSON: "
    '{"patient":{"sex":"male|female|null","age":int|null},'
    '"tests":[{"test":str,"value":number,"unit":str|null,"ref_range":str|null,"confidence":0-1}]}. '
    "Copy values EXACTLY as printed. Never guess or compute a number. If a value is unreadable, omit that test "
    "or give it a low confidence. ref_range is the range printed on the report, e.g. '12.0 - 15.5'.")


def vision_extract(image_bytes: bytes, media_type: str = "image/png") -> tuple[list[ExtractedTest], Patient]:
    llm = get_llm()
    raw = llm.vision(VISION_SYSTEM, "Extract every test result in this report.", image_bytes, media_type)
    data = extract_json(raw)
    patient = Patient(**{k: v for k, v in (data.get("patient") or {}).items() if v not in (None, "null")})
    tests = [ExtractedTest(**t) for t in data.get("tests", [])]
    return tests, patient


def _merge(ocr_tests: list[ExtractedTest], llm_tests: list[ExtractedTest], warnings: list[str]):
    """Cross-check two independent extractions; disagreement lowers confidence."""
    def key(t): return canonical_key(t.test) or t.test.lower()
    ocr_by = {key(t): t for t in ocr_tests}
    out, seen = [], set()
    for t in llm_tests:
        k = key(t)
        seen.add(k)
        o = ocr_by.get(k)
        if o is None:
            out.append(t)
        elif abs(o.value - t.value) <= 0.01 * max(abs(t.value), 1):
            out.append(t.model_copy(update={"confidence": min(1.0, max(t.confidence, 0.9))}))
        else:
            warnings.append(f"{t.test}: OCR read {o.value} but the vision model read {t.value}. Please confirm.")
            out.append(t.model_copy(update={"confidence": 0.5}))
    for k, o in ocr_by.items():
        if k not in seen:
            out.append(o.model_copy(update={"confidence": min(o.confidence, OCR_CONFIDENCE_CAP)}))
    return out


def extract_from_bytes(filename: str, data: bytes) -> ExtractionResult:
    kind = detect_kind(filename, data)
    warnings: list[str] = []
    if kind == "text":
        text = data.decode("utf-8", "ignore")
        return ExtractionResult(tests=parse_report_text(text), patient=parse_patient(text), source="text")

    if kind == "pdf":
        pages = _pdf_pages_text(data)
        text = "\n".join(pages)
        if len(text.strip()) >= 40:
            tests = parse_report_text(text)
            for i, pg in enumerate(pages, 1):
                pass
            return ExtractionResult(tests=tests, patient=parse_patient(text), source="pdf_text")
        warnings.append("This PDF has no readable text (scanned); OCR was used.")
        images = _pdf_pages_images(data)
    elif kind == "image":
        from PIL import Image
        images = [Image.open(io.BytesIO(data))]
    else:
        return ExtractionResult(tests=[], warnings=["Unsupported file type. Upload a PDF, PNG or JPG."], source="none")

    ocr_text = "\n".join(ocr_image(im) for im in images)
    ocr_tests = [t.model_copy(update={"confidence": min(t.confidence, OCR_CONFIDENCE_CAP)})
                 for t in parse_report_text(ocr_text)]
    patient = parse_patient(ocr_text)
    llm = get_llm()
    if not llm.available:
        warnings.append("Read by OCR only, so please check the values before relying on them.")
        return ExtractionResult(tests=ocr_tests, patient=patient, source="ocr", warnings=warnings)

    llm_tests: list[ExtractedTest] = []
    for im in images:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        for attempt in range(2):
            try:
                t, p = vision_extract(buf.getvalue())
                llm_tests += t
                patient = Patient(sex=p.sex or patient.sex, age=p.age or patient.age)
                break
            except Exception as exc:   # bad JSON, schema failure, or a network/API error
                log.warning("vision extraction attempt %s failed: %s", attempt + 1, exc)
        else:
            warnings.append("The vision model could not be used for one page, so OCR values were used; "
                            "please check the values.")
    merged = _merge(ocr_tests, llm_tests, warnings) if llm_tests else ocr_tests
    return ExtractionResult(tests=merged, patient=patient, source="ocr+vision_llm" if llm_tests else "ocr",
                            warnings=warnings)


def extract_from_text(text: str) -> ExtractionResult:
    return ExtractionResult(tests=parse_report_text(text), patient=parse_patient(text), source="text")
