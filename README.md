# MediLens AI - AI/ML backend (Django REST)

Report analysis (CBC) + medicine identification. This is the AI/ML part only; the frontend calls these endpoints.

## Run
```bash
pip install -r requirements.txt      # and install Tesseract OCR on your machine
python manage.py migrate
python manage.py test                # 60 tests, no network or API key needed
python manage.py runserver
```
Optional real model: `export MEDILENS_LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...` (model via `MEDILENS_LLM_MODEL`).
Without it everything still works using OCR + a deterministic explanation template.

## Endpoints
| Method | URL | Purpose |
|---|---|---|
| POST | `/api/reports/` | multipart `file` (PDF/PNG/JPG) **or** `text`; optional `language` (`en`/`roman_urdu`), `sex`, `age` |
| GET | `/api/reports/<id>/` | fetch result |
| POST | `/api/reports/<id>/confirm/` | JSON `{"tests":[{test,value,unit,ref_range}], "sex":..}` after the user checks values |
| DELETE | `/api/reports/<id>/` | delete all stored data |
| POST | `/api/medicines/identify/` | multipart `image` (JPG/PNG/WEBP); optional `ocr_text`, `language` |
| POST | `/api/medicines/confirm/` | JSON `{"brand","strength","form"}` after the user picks a candidate |
| GET | `/api/health/` | status and whether an LLM is configured |

Report `status`: `completed` | `needs_confirmation` (show `extraction` + `needs_confirmation_because`, then call confirm).
Medicine `status`: `identified` | `needs_confirmation` (show `candidates`) | `retake_photo`.

## How it works (matches the PRD doc)
Report: file -> text/OCR/vision -> Pydantic validation -> name+unit normalisation (LOINC) -> **rule-based flags** against the
report's own printed range -> triage (critical values) -> retrieval -> explanation -> verification -> output.
The LLM never decides normal/abnormal. Verification checks: no diagnosis/dosage language, every number in the text comes
from the data, every abnormal result is mentioned, and (with an LLM) a fact-check pass. If anything fails: one retry, then a safe template.
Medicine: quality check -> OCR/vision -> fuzzy match to drug table -> confidence gate -> general use + same ingredient/strength/form alternatives.

## Safety defaults worth knowing
- Images read by OCR alone are **always** sent back for user confirmation (OCR can misread digits).
- Uploaded files are deleted right after processing (`MEDILENS_DELETE_UPLOADS=true`). Medicine photos are never written to disk.
- No dosage, no "take/stop this", no diagnosis wording; medicine answers always end with "confirm with doctor/pharmacist".

## MUST be completed before real users (placeholders, not clinical truth)
1. `core/data/critical_thresholds.json` - **placeholder** limits, `signed_off_by` is null. Needs a doctor/lab advisor.
2. `core/data/reference_ranges.json` - fallback ranges; entries with `"verified": false` need review.
3. `reports/knowledge/cbc_seed.json` - developer-written seed notes. Run `python manage.py ingest_medlineplus "anemia" "blood count tests"`
   (needs internet; credit MedlinePlus.gov) and have a clinician review. Retrieval is keyword-based for now; swap `rag.retrieve` for Chroma + BGE-M3 later.
4. `medicines/data/drugs_sample.csv` - 17-row **sample** from developer knowledge, not DRAP-verified. Replace with DRAP registry data
   (same columns: brand, ingredient, strength, form, general_use) after confirming permission; point `MEDILENS_DRUG_TABLE` in settings at it.
5. Add authentication, rate limiting, HTTPS, a real `SECRET_KEY`/`DEBUG=False` before deployment.

## Not verified live
The Anthropic provider and the MedlinePlus ingest command were only tested with fakes/sample XML (no API key or internet in the build sandbox).
Roman Urdu quality from a real LLM is untested; the offline template uses Roman Urdu for sentences but English for background notes.

## Evaluating on downloaded datasets

```
python manage.py evaluate_csv "D:\data\anemia.csv" --limit 30 --render text   # fast, no OCR
python manage.py evaluate_csv "D:\data\anemia.csv" --limit 10 --render png    # realistic: OCR on rendered images
python manage.py evaluate_images "D:\data\tablet_packs" --limit 30
```
`evaluate_csv` builds a synthetic lab report per CSV row (the CSV has no printed ranges, so ranges are generated),
runs extraction + analysis, and reports extraction accuracy and flag agreement. The dataset's diagnosis label vs our
hemoglobin flag is a sanity check only, not clinical accuracy. `evaluate_images` reports quality-gate rejections,
OCR readout, label-in-OCR hit rate and pipeline status; matching uses only the 17 sample drugs.
