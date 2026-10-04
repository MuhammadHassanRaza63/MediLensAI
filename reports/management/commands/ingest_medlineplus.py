"""Download MedlinePlus health-topic text into reports/knowledge/ingested.json.

    python manage.py ingest_medlineplus "anemia" "blood count tests" "thrombocytopenia"

Uses the free MedlinePlus Web Service. Per their terms, say the information is from
MedlinePlus.gov and do not use their logo or imply endorsement. Needs internet access.
After ingesting, have a clinician review the text before real use.
"""
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from django.core.management.base import BaseCommand

from reports.services import rag

WS = "https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&rettype=brief&retmax=3&term="


def parse_response(xml_text: str) -> list[dict]:
    """Parse a MedlinePlus Web Service XML reply into entries."""
    root = ET.fromstring(xml_text)
    out = []
    for doc in root.iter("document"):
        fields = {c.get("name"): "".join(c.itertext()) for c in doc.findall("content")}
        title = re.sub(r"<[^>]+>", "", fields.get("title", "")).strip()
        text = re.sub(r"<[^>]+>", " ", fields.get("FullSummary") or fields.get("snippet", ""))
        text = re.sub(r"\s+", " ", text).strip()
        if title and text:
            out.append({"title": title, "text": text, "source_url": doc.get("url")})
    return out


class Command(BaseCommand):
    help = "Ingest MedlinePlus health-topic summaries as RAG knowledge."

    def add_arguments(self, parser):
        parser.add_argument("terms", nargs="+")

    def handle(self, *args, **opts):
        path = Path(rag.KB_DIR) / "ingested.json"
        existing = json.loads(path.read_text())["entries"] if path.exists() else []
        have = {e["source_url"] for e in existing}
        added = 0
        for term in opts["terms"]:
            url = WS + urllib.parse.quote(term)
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    entries = parse_response(resp.read().decode("utf-8"))
            except Exception as exc:
                self.stderr.write(f"{term}: failed ({exc})")
                continue
            for e in entries:
                if e["source_url"] in have:
                    continue
                existing.append({"id": f"mlp-{len(existing) + 1}", "test_key": "ingested", "direction": None,
                                 "status": "ingested", **e})
                have.add(e["source_url"])
                added += 1
        path.write_text(json.dumps({"entries": existing}, indent=2, ensure_ascii=False))
        rag.clear_cache()
        self.stdout.write(f"Added {added} entries (source: MedlinePlus.gov). Review before use.")
