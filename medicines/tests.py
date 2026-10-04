import io
from pathlib import Path
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from core.llm import BaseLLM
from medicines.services import agent
from medicines.services.extract import fields_from_text, parse_form, parse_strengths
from medicines.services.match import alternatives, load_drugs, match_drugs
from medicines.services.quality import assess_image


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def pack_image(lines, blur=0, size=(1200, 800), bg="white") -> bytes:
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    y = 60
    for text, px in lines:
        d.text((60, y), text, fill="black", font=_font(px))
        y += int(px * 1.8)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


PANADOL = [("PANADOL", 120), ("Paracetamol 500 mg", 60), ("Tablets", 60)]


class ParsingTests(SimpleTestCase):
    def test_strength_and_form(self):
        self.assertEqual(parse_strengths("Each tablet contains Ibuprofen 400 mg and Pseudoephedrine 60mg"),
                         ["400 mg", "60 mg"])
        self.assertEqual(parse_strengths("Paracetamol 0.5 g"), ["0.5 g"])
        self.assertEqual(parse_form("PANADOL 500 mg TABLETS"), "tablet")
        self.assertEqual(fields_from_text("Risek 40 mg capsules").strengths, ["40 mg"])


class MatchTests(SimpleTestCase):
    def test_brand_with_ocr_noise(self):
        c = match_drugs(fields_from_text("PANAD0L\nParacetamol 500 mg\nTablets"), "PANAD0L\nParacetamol 500 mg\nTablets")
        self.assertEqual(c[0]["brand"], "Panadol")

    def test_combination_product_picks_correct_strength_variant(self):
        txt = "ARINAC FORTE\nIbuprofen 400 mg\nPseudoephedrine HCl 60 mg\nTablets"
        c = match_drugs(fields_from_text(txt), txt)
        self.assertEqual(c[0]["brand"], "Arinac Forte")
        self.assertGreater(c[0]["score"], c[1]["score"] if len(c) > 1 else 0)

    def test_strength_conflict_lowers_score(self):
        ok = match_drugs(fields_from_text("Panadol 500 mg tablets"), "Panadol 500 mg tablets")[0]
        bad = match_drugs(fields_from_text("Panadol 650 mg tablets"), "Panadol 650 mg tablets")[0]
        self.assertEqual(bad["strength_state"], "conflict")
        self.assertLess(bad["score"], ok["score"] - 20)

    def test_generic_name_only_is_ingredient_only(self):
        c = match_drugs(fields_from_text("Metformin 500 mg tablets"), "Metformin 500 mg tablets")[0]
        self.assertEqual(c["ingredient"], "metformin")
        self.assertTrue(c["ingredient_only"])
        self.assertLess(c["score"], 90)

    def test_alias_acetaminophen(self):
        c = match_drugs(fields_from_text("Acetaminophen 500 mg tablets"), "Acetaminophen 500 mg tablets")
        self.assertEqual(c[0]["ingredient"], "paracetamol")

    def test_unknown_text_matches_nothing(self):
        self.assertEqual(match_drugs(fields_from_text("Hello world 12 kg"), "Hello world 12 kg"), [])

    def test_alternatives_are_same_ingredient_strength_form_only(self):
        pan = next(r for r in load_drugs() if r["brand"] == "Panadol")
        alts = alternatives(pan)
        self.assertEqual([a["brand"] for a in alts], ["Calpol"])
        for a in alts:
            self.assertEqual((a["ingredient"], a["strength"], a["form"]), (pan["ingredient"], pan["strength"], pan["form"]))
        risek20 = next(r for r in load_drugs() if r["brand"] == "Risek" and r["strength"] == "20 mg")
        self.assertEqual([a["brand"] for a in alternatives(risek20)], ["Losec"])   # not Risek 40 mg


class QualityTests(SimpleTestCase):
    def test_sharp_ok_blurry_rejected_dark_rejected(self):
        self.assertTrue(assess_image(pack_image(PANADOL))["ok"])
        blurry = assess_image(pack_image(PANADOL, blur=12))
        self.assertFalse(blurry["ok"])
        self.assertTrue(any("blurry" in i for i in blurry["issues"]))
        dark = assess_image(pack_image([("x", 10)], bg=(5, 5, 5)))
        self.assertFalse(dark["ok"])
        tiny = assess_image(pack_image(PANADOL, size=(300, 200)))
        self.assertTrue(any("small" in i for i in tiny["issues"]))


class AgentTests(SimpleTestCase):
    def test_clear_photo_is_identified_with_guardrails(self):
        out = agent.identify(pack_image(PANADOL))
        self.assertEqual(out["status"], "identified", out)
        res = out["result"]
        self.assertEqual((res["brand"], res["active_ingredient"], res["strength"]), ("Panadol", "paracetamol", "500 mg"))
        self.assertEqual([a["brand"] for a in res["alternatives"]], ["Calpol"])
        self.assertIn("pharmacist", res["disclaimer"])
        self.assertNotIn("dose", (res["general_use"] or "").lower())

    def test_blurry_photo_asks_for_retake(self):
        out = agent.identify(pack_image(PANADOL, blur=12))
        self.assertEqual(out["status"], "retake_photo")
        self.assertNotIn("result", out)

    def test_strength_conflict_needs_confirmation(self):
        out = agent.identify(None, ocr_text="PANADOL\nParacetamol 650 mg\nTablets")
        self.assertEqual(out["status"], "needs_confirmation")
        self.assertNotIn("result", out)
        self.assertEqual(out["candidates"][0]["brand"], "Panadol")

    def test_unrecognised_text_retake(self):
        out = agent.identify(None, ocr_text="Some random soap bar 100 g")
        self.assertEqual(out["status"], "retake_photo")

    def test_generic_only_pack_needs_confirmation(self):
        out = agent.identify(None, ocr_text="Metformin 500 mg tablets")
        self.assertEqual(out["status"], "needs_confirmation")

    def test_roman_urdu_disclaimer(self):
        out = agent.identify(None, ocr_text="PANADOL Paracetamol 500 mg Tablets", language="roman_urdu")
        self.assertIn("pharmacist", out["disclaimer"])
        self.assertIn("confirm", out["disclaimer"])

    def test_policy_guard_removes_unsafe_general_use(self):
        card = agent._result_card({"brand": "X", "ingredient": "y", "strength": "1 mg", "form": "tablet",
                                   "general_use": "Take this tablet twice daily."}, "en")
        self.assertIsNone(card["general_use"])

    def test_confirm(self):
        out = agent.confirm("risek", "40 mg")
        self.assertEqual(out["result"]["strength"], "40 mg")
        self.assertEqual(out["result"]["alternatives"], [])
        amb = agent.confirm("Risek")
        self.assertEqual(amb["ambiguous_strengths"], ["20 mg", "40 mg"])
        self.assertIsNone(agent.confirm("Nonexistent"))

    def test_vision_fields_are_used_and_failures_fall_back(self):
        class V(BaseLLM):
            available = True
            def vision(self, *a, **k):
                return ('{"brand":"Brufen","generic_names":["Ibuprofen"],"strengths":["400 mg"],'
                        '"form":"tablet","confidence":0.9}')
        with mock.patch("medicines.services.extract.get_llm", return_value=V()):
            out = agent.identify(pack_image([("brufen", 100)]))
        self.assertEqual(out["extracted"]["brand"], "Brufen")
        self.assertEqual(out["candidates"][0]["brand"], "Brufen")

        class Boom(V):
            def vision(self, *a, **k):
                raise ConnectionError("down")
        with mock.patch("medicines.services.extract.get_llm", return_value=Boom()):
            out = agent.identify(pack_image(PANADOL))
        self.assertEqual(out["status"], "identified")
        self.assertTrue(any("could not be used" in n for n in out["notes"]))


class MedicineApiTests(TestCase):
    def test_identify_upload_and_confirm(self):
        f = SimpleUploadedFile("p.jpg", pack_image(PANADOL), content_type="image/jpeg")
        r = self.client.post("/api/medicines/identify/", {"image": f})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["status"], "identified")
        c = self.client.post("/api/medicines/confirm/", {"brand": "Brufen"}, content_type="application/json")
        self.assertEqual(c.status_code, 200)
        self.assertEqual(c.json()["result"]["active_ingredient"], "ibuprofen")

    def test_errors(self):
        self.assertEqual(self.client.post("/api/medicines/identify/", {}).status_code, 400)
        bad = SimpleUploadedFile("p.gif", b"GIF89a", content_type="image/gif")
        self.assertEqual(self.client.post("/api/medicines/identify/", {"image": bad}).status_code, 415)
        corrupt = SimpleUploadedFile("p.jpg", b"not an image", content_type="image/jpeg")
        self.assertEqual(self.client.post("/api/medicines/identify/", {"image": corrupt}).status_code, 422)
        self.assertEqual(self.client.post("/api/medicines/confirm/", {}, content_type="application/json").status_code, 400)
        self.assertEqual(self.client.post("/api/medicines/confirm/", {"brand": "Zzz"}, content_type="application/json").status_code, 404)

    def test_health(self):
        self.assertEqual(self.client.get("/api/health/").json(), {"status": "ok", "llm_available": False})


class EvaluateImagesCommandTests(SimpleTestCase):
    def test_runs_on_folder_with_folder_labels(self):
        import os
        import tempfile
        from django.core.management import call_command
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "Panadol"
            sub.mkdir()
            (sub / "a.jpg").write_bytes(pack_image([("Panadol", 80), ("Paracetamol 500 mg", 40), ("Tablets", 40)]))
            out = io.StringIO()
            call_command("evaluate_images", d, "--limit", "5", "--out", os.path.join(d, "r.csv"), stdout=out)
            text = out.getvalue()
            self.assertIn("Images processed: 1", text)
            self.assertTrue(os.path.exists(os.path.join(d, "r.csv")))
