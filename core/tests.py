from django.test import SimpleTestCase, override_settings

from core.llm import MockLLM, extract_json, get_llm, reset_llm_cache
from core.safety import check_policy, is_safe


class PolicyTests(SimpleTestCase):
    def test_blocks_diagnosis_dosage_and_start_stop(self):
        bad = [
            "You have anemia.",
            "You are suffering from a blood disorder.",
            "Aap ko anemia hai.",
            "Take 500 mg twice daily.",
            "Take this medicine after food.",
            "You should take iron tablets.",
            "Stop taking your medicine today.",
            "Ye dawa lein.",
            "Tablet chhor dein.",
            "The dose is 2 tablets.",
        ]
        for text in bad:
            self.assertFalse(is_safe(text), text)

    def test_allows_normal_explanations(self):
        good = [
            "Hemoglobin: 10.2 g/dL (range: 12 - 15.5) - lower than the range.",
            "It is not a diagnosis and not medical advice. Please discuss your results with a doctor.",
            "Please show this report to a doctor, who can read it together with your symptoms.",
            "Used to reduce fever and relieve mild to moderate pain.",
            "Agla qadam: Meherbani kar ke ye report doctor ko dikhayein.",
            "Same active ingredient, strength and dosage form only. Ask your pharmacist.",
        ]
        for text in good:
            self.assertEqual(check_policy(text), [], text)


class LLMTests(SimpleTestCase):
    def test_default_is_mock_and_unavailable(self):
        reset_llm_cache()
        self.assertIsInstance(get_llm(), MockLLM)
        self.assertFalse(get_llm().available)

    @override_settings(MEDILENS_LLM_PROVIDER="anthropic")
    def test_anthropic_without_key_falls_back_to_mock(self):
        import os
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        reset_llm_cache()
        try:
            self.assertFalse(get_llm().available)
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old
            reset_llm_cache()

    def test_extract_json_handles_fences_and_prose(self):
        self.assertEqual(extract_json('Here:\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('ok {"a": [1,2]} done'), {"a": [1, 2]})
        with self.assertRaises(ValueError):
            extract_json("no json here")
