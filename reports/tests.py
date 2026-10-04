import io
import shutil
import tempfile
from pathlib import Path
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from PIL import Image, ImageDraw, ImageFont

from core.llm import BaseLLM
from reports.models import ReportUpload
from reports.services import rag
from reports.services.analyze import analyze_all, apply_triage, pattern_hints
from reports.services.explain import template_explanation
from reports.services.extract import extract_from_bytes, extract_from_text, parse_report_text
from reports.services.normalize import canonical_key, normalize_test, parse_range
from reports.services.pipeline import run_analysis
from reports.services.verify import verify_explanation

CBC_TEXT = """CITY LAB - Complete Blood Count
Patient Name: Test Patient   Age: 34   Sex: Female
Test            Result    Unit        Reference Range
Hemoglobin (Hb)   10.2 L  g/dL        12.0 - 15.5
Total RBC Count   4.1     million/cumm 3.8 - 4.8
MCV               72      fL          80 - 100
WBC Count         7,500   /cumm       4000 - 11000
Platelet Count    210     x10^3/uL    150 - 400
Date: 12-10-2025
Page 1 of 1
"""

CRITICAL_TEXT = "Sex: Male\nHemoglobin 6.1 g/dL 13.5 - 17.5\nPlatelet Count 15 x10^3/uL 150 - 400\n"


class NormalizeTests(SimpleTestCase):
    def test_names(self):
        for printed, key in [("Hemoglobin (Hb)", "hemoglobin"), ("HGB", "hemoglobin"), ("Haemoglobin", "hemoglobin"),
                             ("Platelet Count", "platelets"), ("T.L.C", "wbc"), ("RDW-CV", "rdw"),
                             ("PCV", "hematocrit"), ("Hemoglobn", "hemoglobin")]:
            self.assertEqual(canonical_key(printed), key, printed)
        self.assertIsNone(canonical_key("Blood Urea"))

    def test_units_scale_value_and_printed_range_together(self):
        t = normalize_test("WBC", 7500, "/cumm", "4000 - 11000")
        self.assertEqual((t.value, t.ref_low, t.ref_high), (7.5, 4.0, 11.0))
        t = normalize_test("Hb", 102, "g/L", "120 - 155")
        self.assertEqual((t.value, t.ref_low, t.ref_high), (10.2, 12.0, 15.5))
        t = normalize_test("Platelets", 210000, None, None)
        self.assertEqual(t.value, 210.0)

    def test_parse_range(self):
        self.assertEqual(parse_range("12.0 - 15.5"), (12.0, 15.5))
        self.assertEqual(parse_range("4,000-11,000"), (4000.0, 11000.0))
        self.assertEqual(parse_range("15 - 12"), (None, None))
        self.assertEqual(parse_range(None), (None, None))


class ParserTests(SimpleTestCase):
    def test_parses_cbc_lines_and_ignores_junk(self):
        tests = {t.test: t for t in parse_report_text(CBC_TEXT)}
        self.assertEqual(set(tests), {"Hemoglobin (Hb)", "Total RBC Count", "MCV", "WBC Count", "Platelet Count"})
        self.assertEqual(tests["WBC Count"].value, 7500)
        self.assertEqual(tests["Hemoglobin (Hb)"].flag_raw, "L")
        self.assertEqual(tests["Hemoglobin (Hb)"].ref_range, "12.0 - 15.5")
        ex = extract_from_text(CBC_TEXT)
        self.assertEqual((ex.patient.sex, ex.patient.age), ("female", 34))


class AnalyzeTests(SimpleTestCase):
    def _run(self, text, sex=None):
        ex = extract_from_text(text)
        norm = [normalize_test(t.test, t.value, t.unit, t.ref_range, t.confidence, t.flag_raw) for t in ex.tests]
        res = analyze_all(norm, sex or ex.patient.sex)
        alerts = apply_triage(res)
        return {r.key: r for r in res}, alerts, res

    def test_flags_use_report_range(self):
        r, alerts, _ = self._run(CBC_TEXT)
        self.assertEqual(r["hemoglobin"].flag, "Low")
        self.assertEqual(r["hemoglobin"].ref_source, "report")
        self.assertEqual(r["mcv"].flag, "Low")
        self.assertEqual(r["platelets"].flag, "Normal")
        self.assertEqual(r["wbc"].flag, "Normal")
        self.assertEqual(alerts, [])

    def test_boundary_values_are_normal(self):
        r, _, _ = self._run("Hemoglobin 12.0 g/dL 12.0 - 15.5\nPlatelet Count 400 x10^3/uL 150 - 400\n")
        self.assertEqual(r["hemoglobin"].flag, "Normal")
        self.assertEqual(r["platelets"].flag, "Normal")

    def test_critical_values(self):
        r, alerts, _ = self._run(CRITICAL_TEXT)
        self.assertEqual(r["hemoglobin"].flag, "Critical")
        self.assertEqual(r["hemoglobin"].direction, "low")
        self.assertEqual(r["platelets"].flag, "Critical")
        self.assertEqual(len(alerts), 2)

    def test_fallback_range_when_none_printed(self):
        r, _, _ = self._run("Hemoglobin 11.0 g/dL\n", sex="female")
        self.assertEqual(r["hemoglobin"].ref_source, "fallback")
        self.assertEqual(r["hemoglobin"].flag, "Low")
        r, _, _ = self._run("Hemoglobin 11.0 g/dL\n", sex="male")
        self.assertEqual(r["hemoglobin"].flag, "Low")
        r, _, _ = self._run("Hemoglobin 12.5 g/dL\n", sex=None)       # wide range when sex unknown
        self.assertEqual(r["hemoglobin"].flag, "Normal")
        self.assertTrue(any("Sex not given" in n for n in r["hemoglobin"].notes))

    def test_unknown_test_without_range_is_not_assessed(self):
        norm = [normalize_test("Blood Urea", 60, "mg/dL", None)]
        self.assertEqual(analyze_all(norm, "male")[0].flag, "Not assessed")

    def test_report_flag_disagreement_is_noted(self):
        norm = [normalize_test("Hemoglobin", 13.0, "g/dL", "12.0 - 15.5", flag_raw="L")]
        res = analyze_all(norm, "female")
        self.assertEqual(res[0].flag, "Normal")
        self.assertTrue(any("printed" in n for n in res[0].notes))

    def test_pattern_hints(self):
        _, _, res = self._run(CBC_TEXT)
        hints = pattern_hints(res)
        self.assertEqual(len(hints), 1)
        self.assertIn("microcytic", hints[0])


class PipelineTests(SimpleTestCase):
    def test_template_passes_verification_in_both_languages(self):
        for lang in ("en", "roman_urdu"):
            out = run_analysis(extract_from_text(CBC_TEXT), lang)
            self.assertTrue(out["verification"]["passed"], out["verification"])
            self.assertEqual(out["explanation_source"], "template")
            self.assertIn("10.2", out["explanation"])
            self.assertIsNone(out["urgent_warning"])

    def test_critical_warning_comes_first(self):
        out = run_analysis(extract_from_text(CRITICAL_TEXT), "en")
        self.assertTrue(out["explanation"].startswith(out["urgent_warning"]))
        self.assertEqual(len(out["alerts"]), 2)
        self.assertTrue(any("sign-off" in n for n in out["notices"]))

    def test_all_normal_report(self):
        out = run_analysis(extract_from_text("Hemoglobin 13.5 g/dL 12 - 15.5\nWBC 6.0 x10^3/uL 4 - 11\n"), "en")
        self.assertEqual(out["alerts"], [])
        self.assertIn("0 outside it", out["explanation"])
        self.assertTrue(out["verification"]["passed"])

    def test_verifier_catches_wrong_number_missing_result_and_diagnosis(self):
        out = run_analysis(extract_from_text(CBC_TEXT), "en")
        results, hints = out["results"], out["hints"]
        good = template_explanation(results, hints, {}, "en")
        self.assertTrue(verify_explanation(good, results, hints, {})["passed"])
        wrong = good.replace("10.2", "9.2")
        v = verify_explanation(wrong, results, hints, {})
        self.assertFalse(v["passed"])
        self.assertTrue(any(c["name"] == "numbers_match_data" and not c["passed"] for c in v["checks"]))
        self.assertTrue(any(c["name"] == "abnormal_results_covered" and not c["passed"] for c in v["checks"]))
        diag = good + "\nYou have iron deficiency anemia."
        v = verify_explanation(diag, results, hints, {})
        self.assertTrue(any(c["name"] == "policy" and not c["passed"] for c in v["checks"]))

    def test_seed_notes_are_cited_and_flagged(self):
        out = run_analysis(extract_from_text(CBC_TEXT), "en")
        ids = {c["id"] for c in out["citations"]}
        self.assertIn("seed-hemoglobin-low", ids)
        self.assertTrue(any("seed" in n for n in out["notices"]))


class FakeLLM(BaseLLM):
    available = True

    def __init__(self, drafts, judge='{"unsupported": []}'):
        self.drafts, self.judge, self.calls = list(drafts), judge, 0

    def complete(self, system, user, max_tokens=1200):
        if "fact-checker" in system:
            return self.judge
        self.calls += 1
        return self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]


class LLMPathTests(SimpleTestCase):
    GOOD = ("Overview: 5 results, 2 outside the range.\n- Hemoglobin: 10.2 g/dL (range: 12 - 15.5), lower than the range.\n"
            "- MCV: 72 fL (range: 80 - 100), lower than the range.\nPlease show this report to a doctor.")

    def _run(self, llm):
        with mock.patch("reports.services.pipeline.get_llm", return_value=llm):
            return run_analysis(extract_from_text(CBC_TEXT), "en")

    def test_good_llm_draft_is_used(self):
        out = self._run(FakeLLM([self.GOOD]))
        self.assertEqual(out["explanation_source"], "llm")
        self.assertTrue(out["verification"]["passed"])

    def test_bad_draft_is_regenerated_once_then_template(self):
        bad = self.GOOD.replace("10.2", "9.9")
        llm = FakeLLM([bad])
        out = self._run(llm)
        self.assertEqual(out["explanation_source"], "template_fallback")
        self.assertEqual(llm.calls, 2)
        self.assertIn("10.2", out["explanation"])
        self.assertNotIn("9.9", out["explanation"])

    def test_diagnosis_from_llm_never_reaches_user(self):
        out = self._run(FakeLLM([self.GOOD + "\nYou have anemia."]))
        self.assertNotIn("You have anemia", out["explanation"])
        self.assertEqual(out["explanation_source"], "template_fallback")

    def test_judge_flagging_claim_forces_fallback(self):
        out = self._run(FakeLLM([self.GOOD], judge='{"unsupported": ["Dengue is likely."]}'))
        self.assertEqual(out["explanation_source"], "template_fallback")

    def test_llm_failure_falls_back(self):
        class Boom(FakeLLM):
            def complete(self, *a, **k):
                raise RuntimeError("network down")
        out = self._run(Boom([self.GOOD]))
        self.assertIn(out["explanation_source"], ("template_fallback", "template"))
        self.assertTrue(out["verification"]["passed"])


class RagTests(SimpleTestCase):
    def test_retrieve_prefers_matching_direction(self):
        notes = rag.retrieve("hemoglobin", "low", "Hemoglobin")
        self.assertEqual(notes[0]["id"], "seed-hemoglobin-low")
        self.assertEqual(rag.retrieve("hemoglobin", "high")[0]["id"], "seed-hemoglobin-high")
        self.assertEqual(rag.retrieve("nonsense", "low"), [])

    def test_medlineplus_xml_parser(self):
        from reports.management.commands.ingest_medlineplus import parse_response
        xml = ('<nlmSearchResult><list><document url="https://medlineplus.gov/anemia.html" rank="0">'
               '<content name="title">&lt;span&gt;Anemia&lt;/span&gt;</content>'
               '<content name="FullSummary">&lt;p&gt;Anemia is a problem of not having enough healthy red blood cells.&lt;/p&gt;</content>'
               '</document></list></nlmSearchResult>')
        entries = parse_response(xml)
        self.assertEqual(entries[0]["title"], "Anemia")
        self.assertIn("healthy red blood cells", entries[0]["text"])
        self.assertEqual(entries[0]["source_url"], "https://medlineplus.gov/anemia.html")


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render_report_png(text=CBC_TEXT) -> bytes:
    lines = text.strip().splitlines()
    img = Image.new("RGB", (1500, 60 + 50 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((40, 30 + 50 * i), line, fill="black", font=_font(30))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def render_report_pdf(text=CBC_TEXT) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in text.strip().splitlines():
        c.drawString(40, y, line)
        y -= 18
    c.save()
    return buf.getvalue()


class ExtractFileTests(SimpleTestCase):
    def test_digital_pdf_text_path(self):
        ex = extract_from_bytes("r.pdf", render_report_pdf())
        self.assertEqual(ex.source, "pdf_text")
        self.assertEqual(len(ex.tests), 5)
        self.assertTrue(all(t.confidence >= 0.8 for t in ex.tests))

    def test_scanned_pdf_goes_through_ocr(self):
        img = Image.open(io.BytesIO(render_report_png())).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "PDF", resolution=150)
        ex = extract_from_bytes("scan.pdf", buf.getvalue())
        self.assertEqual(ex.source, "ocr")
        self.assertTrue(any("scanned" in w for w in ex.warnings))
        self.assertTrue(any(t.test.lower().startswith("hemoglobin") for t in ex.tests))
        self.assertTrue(all(t.confidence <= 0.75 for t in ex.tests))

    def test_image_ocr_path_is_capped_below_threshold(self):
        ex = extract_from_bytes("r.png", render_report_png())
        self.assertEqual(ex.source, "ocr")
        self.assertTrue(ex.tests, "OCR found nothing")
        self.assertTrue(all(t.confidence <= 0.75 for t in ex.tests))
        hb = [t for t in ex.tests if t.test.lower().startswith("hemoglobin")]
        self.assertTrue(hb and abs(hb[0].value - 10.2) < 0.01, [t.model_dump() for t in ex.tests])


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ReportApiTests(TestCase):
    def tearDown(self):
        shutil.rmtree(tempfile.gettempdir() + "/_unused", ignore_errors=True)

    def test_text_upload_completes(self):
        r = self.client.post("/api/reports/", {"text": CBC_TEXT, "language": "roman_urdu"})
        self.assertEqual(r.status_code, 201)
        body = r.json()
        self.assertEqual(body["status"], "completed")
        self.assertTrue(body["analysis"]["verification"]["passed"])
        self.assertEqual(body["analysis"]["language"], "roman_urdu")
        again = self.client.get(f"/api/reports/{body['id']}/").json()
        self.assertEqual(again["status"], "completed")

    def test_pdf_upload_completes_and_file_is_not_kept(self):
        f = SimpleUploadedFile("cbc.pdf", render_report_pdf(), content_type="application/pdf")
        r = self.client.post("/api/reports/", {"file": f})
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(r.json()["status"], "completed")
        rep = ReportUpload.objects.get(pk=r.json()["id"])
        self.assertFalse(rep.file)
        self.assertTrue(rep.file_deleted)

    def test_image_upload_asks_for_confirmation_then_confirm_runs_analysis(self):
        f = SimpleUploadedFile("cbc.png", render_report_png(), content_type="image/png")
        r = self.client.post("/api/reports/", {"file": f})
        self.assertEqual(r.status_code, 201, r.content)
        body = r.json()
        self.assertEqual(body["status"], "needs_confirmation")
        self.assertNotIn("analysis", body)
        tests = [{"test": "Hemoglobin", "value": 10.2, "unit": "g/dL", "ref_range": "12.0 - 15.5"},
                 {"test": "MCV", "value": 72, "unit": "fL", "ref_range": "80 - 100"}]
        c = self.client.post(f"/api/reports/{body['id']}/confirm/", {"tests": tests, "sex": "female"},
                             content_type="application/json")
        self.assertEqual(c.status_code, 200, c.content)
        out = c.json()
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["analysis"]["patient"]["sex"], "female")
        self.assertEqual({r_["flag"] for r_ in out["analysis"]["results"]}, {"Low"})

    def test_validation_and_errors(self):
        self.assertEqual(self.client.post("/api/reports/", {}).status_code, 400)
        bad = SimpleUploadedFile("x.exe", b"MZ....", content_type="application/octet-stream")
        self.assertEqual(self.client.post("/api/reports/", {"file": bad}).status_code, 415)
        corrupt = SimpleUploadedFile("c.pdf", b"%PDF-1.4 not really", content_type="application/pdf")
        self.assertEqual(self.client.post("/api/reports/", {"file": corrupt}).status_code, 422)
        r = self.client.post("/api/reports/", {"text": "nothing useful here"})
        self.assertEqual(r.json()["status"], "needs_confirmation")
        missing = "00000000-0000-0000-0000-000000000000"
        self.assertEqual(self.client.get(f"/api/reports/{missing}/").status_code, 404)
        r = self.client.post(f"/api/reports/{r.json()['id']}/confirm/", {"tests": [{"test": "Hb", "value": "abc"}]},
                             content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_delete_removes_everything(self):
        rid = self.client.post("/api/reports/", {"text": CBC_TEXT}).json()["id"]
        self.assertEqual(self.client.delete(f"/api/reports/{rid}/").status_code, 204)
        self.assertEqual(self.client.get(f"/api/reports/{rid}/").status_code, 404)


class VisionExtractionTests(SimpleTestCase):
    """Vision LLM path with a fake model: cross-check against OCR, and failure fallbacks."""

    class Vision(BaseLLM):
        available = True

        def __init__(self, reply=None, error=None):
            self.reply, self.error = reply, error

        def vision(self, system, user, image_bytes, media_type="image/png", max_tokens=1500):
            if self.error:
                raise self.error
            return self.reply

    def _extract(self, llm):
        with mock.patch("reports.services.extract.get_llm", return_value=llm):
            return extract_from_bytes("r.png", render_report_png())

    def test_agreement_raises_confidence_disagreement_flags_it(self):
        reply = ('{"patient":{"sex":"female","age":34},"tests":['
                 '{"test":"Hemoglobin","value":10.2,"unit":"g/dL","ref_range":"12.0 - 15.5","confidence":0.9},'
                 '{"test":"MCV","value":77,"unit":"fL","ref_range":"80 - 100","confidence":0.9}]}')
        ex = self._extract(self.Vision(reply))
        by = {t.test.lower(): t for t in ex.tests}
        self.assertEqual(ex.source, "ocr+vision_llm")
        self.assertGreaterEqual(by["hemoglobin"].confidence, 0.9)
        self.assertEqual(by["mcv"].confidence, 0.5)               # OCR said 72, model said 77
        self.assertTrue(any("MCV" in w for w in ex.warnings))

    def test_network_error_falls_back_to_ocr_and_asks_for_confirmation(self):
        ex = self._extract(self.Vision(error=ConnectionError("down")))
        self.assertEqual(ex.source, "ocr")
        self.assertTrue(ex.tests)
        self.assertTrue(any("check the values" in w for w in ex.warnings))

    def test_garbage_json_falls_back(self):
        ex = self._extract(self.Vision("sorry, I cannot read this"))
        self.assertEqual(ex.source, "ocr")


class GarbledUnitTests(SimpleTestCase):
    """OCR often mangles 'x10^3/uL'. That must never turn a normal count into a false critical value."""

    def test_garbled_unit_with_printed_range_keeps_scale(self):
        from reports.services.normalize import normalize_test
        for unit in ("/uL", "%", "x10*3/uL", None):
            t = normalize_test("WBC count", 4.6, unit, "4.5-11.0", 0.7)
            self.assertAlmostEqual(t.value, 4.6, places=2, msg=unit)

    def test_unit_regex_tolerates_ocr_junk(self):
        from reports.services.extract import parse_report_text
        r = parse_report_text("WBC count 4.6 x10*3/uL 4.5-11.0\nPlatelets 358 x10*%3/uL 150-400\n")
        units = {t.test.lower(): t.unit for t in r}
        self.assertTrue(all(u and "10^3" in u for u in units.values()), units)
