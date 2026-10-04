"""Run the medicine pipeline over a folder of pack photos and report what happens.

    python manage.py evaluate_images "D:\\data\\tablet_pack_images" --limit 30
    python manage.py evaluate_images "D:\\data\\images" --labels labels.csv   # columns: filename,label

Labels come from --labels, else from the parent folder name (if the dataset is organised as folder-per-medicine).
This measures: quality-gate behaviour, how much text OCR reads, whether the label text appears in the OCR text,
and the pipeline status. It cannot measure identification accuracy unless the drugs are in medicines/data/*.csv.
"""
import csv
import io
import random
from collections import Counter
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from PIL import Image
from rapidfuzz import fuzz

from medicines.services import agent

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GENERIC_DIRS = {"images", "image", "train", "test", "val", "valid", "validation", "data", "dataset", "jpg", "jpeg", "png"}


def downscale(path: Path, max_side: int) -> bytes:
    img = Image.open(path)
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def label_in_text(label: str, text: str) -> bool:
    label = (label or "").strip().lower()
    if len(label) < 4:
        return False
    return fuzz.partial_ratio(label, (text or "").lower()) >= 80


class Command(BaseCommand):
    help = "Evaluate the medicine identification pipeline on a folder of pack photos."

    def add_arguments(self, parser):
        parser.add_argument("folder")
        parser.add_argument("--limit", type=int, default=30)
        parser.add_argument("--labels", help="CSV with columns filename,label")
        parser.add_argument("--max-side", type=int, default=1800)
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--out", default="eval_images_results.csv")

    def handle(self, *args, **o):
        root = Path(o["folder"])
        if not root.is_dir():
            raise CommandError(f"Folder not found: {root}")
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
        if not files:
            raise CommandError("No jpg/png/webp images found. Check the folder path (or unzip the dataset first).")
        labels = {}
        if o["labels"]:
            with open(o["labels"], newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    labels[Path(r["filename"]).name.lower()] = r["label"]
        random.Random(o["seed"]).shuffle(files)
        files = files[: o["limit"]]
        self.stdout.write(f"Found images under {root}; evaluating {len(files)} (random sample).")

        rows, statuses, hits, labelled = [], Counter(), 0, 0
        for p in files:
            label = labels.get(p.name.lower())
            if label is None and p.parent != root and p.parent.name.lower() not in GENERIC_DIRS:
                label = p.parent.name
            try:
                res = agent.identify(downscale(p, o["max_side"]), "image/jpeg")
            except Exception as e:  # keep going on a bad file
                rows.append({"file": p.name, "label": label or "", "status": f"error: {e}"})
                statuses["error"] += 1
                continue
            text = res.get("ocr_text", "")
            best = (res.get("candidates") or [{}])[0]
            hit = label_in_text(label, text) if label else None
            if label:
                labelled += 1
                hits += bool(hit)
            statuses[res["status"]] += 1
            rows.append({"file": p.name, "label": label or "", "status": res["status"],
                         "quality_ok": res["quality"].get("ok"), "issues": "; ".join(res["quality"].get("issues", [])),
                         "ocr_chars": len(text), "label_in_ocr": hit if hit is not None else "",
                         "best_match": best.get("brand", ""), "score": best.get("score", ""),
                         "ocr_snippet": " ".join(text.split())[:120]})

        n = len(rows)
        blocked = sum(1 for r in rows if r.get("quality_ok") is False)
        readable = sum(1 for r in rows if (r.get("ocr_chars") or 0) >= 15)
        self.stdout.write(f"\nImages processed: {n}")
        self.stdout.write(f"Rejected by quality gate (blur/dark/small): {blocked}/{n}")
        self.stdout.write(f"OCR read some text (>=15 chars): {readable}/{n}")
        self.stdout.write("Pipeline status: " + ", ".join(f"{k}={v}" for k, v in statuses.items()))
        if labelled:
            self.stdout.write(f"Label text found in OCR output: {hits}/{labelled} ({100*hits/labelled:.0f}%)")
        else:
            self.stdout.write("No labels found (use --labels labels.csv, or a folder-per-medicine layout).")
        self.stdout.write("Note: matching is against medicines/data/drugs_sample.csv (17 sample rows), so most photos "
                          "of other brands will correctly end as retake_photo / needs_confirmation.")
        if rows:
            cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
            with open(o["out"], "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
            self.stdout.write(f"Per-image results written to {o['out']}")
