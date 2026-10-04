"""Turn rows of a public CBC CSV into synthetic lab reports and measure the pipeline on them.

    python manage.py evaluate_csv path/to/data.csv --limit 30 --render text|pdf|png

What it checks
  * extraction: is every printed value read back correctly (text/PDF/OCR path)?
  * flags: does the pipeline's Low/Normal/High agree with comparing the value to the printed range directly?
  * sanity: how do the dataset's own labels (e.g. anemia diagnosis) line up with our hemoglobin flag?
It does NOT prove clinical correctness: the CSV has no printed lab ranges, so we print standard ones.
Rows that need confirmation in the real API (OCR) are analysed anyway, to measure the rules.
"""
import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from core.synthetic import render_pdf, render_png
from reports.services.analyze import load_reference_ranges
from reports.services.explain import fmt
from reports.services.extract import extract_from_bytes, extract_from_text
from reports.services.normalize import canonical_key
from reports.services.pipeline import run_analysis

PRINT_NAMES = {"hemoglobin": "Hemoglobin (Hb)", "rbc": "Total RBC Count", "hematocrit": "Hematocrit (PCV)",
               "mcv": "MCV", "mch": "MCH", "mchc": "MCHC", "rdw": "RDW-CV", "wbc": "WBC Count",
               "platelets": "Platelet Count"}
BASE_UNITS = {"hemoglobin": "g/dL", "rbc": "million/cumm", "hematocrit": "%", "mcv": "fL", "mch": "pg",
              "mchc": "g/dL", "rdw": "%"}
LABEL_HINTS = ("diagnos", "anemi", "label", "class", "target", "status", "severity", "type", "result")


def clean_header(h: str) -> str:
    h = re.sub(r"\(.*?\)", " ", h or "")
    h = h.split("/")[0]
    return re.sub(r"[^A-Za-z% ]", " ", h).strip()


def to_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> list[list[str]]:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as fh:
                return list(csv.reader(fh))
        except UnicodeDecodeError:
            continue
    raise CommandError("Could not read the CSV (unknown encoding).")


def map_header(row: list[str]) -> dict:
    cols, sex, age, label = {}, None, None, None
    for i, h in enumerate(row):
        c = clean_header(h)
        low = c.lower()
        if not c:
            continue
        key = canonical_key(c) if c.lower() not in ("sex", "age") else None
        if key in PRINT_NAMES or key in ("wbc", "platelets"):
            cols.setdefault(key, i)
        elif low in ("sex", "gender"):
            sex = i
        elif low == "age":
            age = i
        elif label is None and any(t in low for t in LABEL_HINTS):
            label = i
    return {"cols": cols, "sex": sex, "age": age, "label": label}


def parse_sex(v):
    v = (v or "").strip().lower()
    if v in ("m", "male"):
        return "male"
    if v in ("f", "female"):
        return "female"
    return None


class Command(BaseCommand):
    help = "Evaluate the report pipeline on a public CBC CSV (e.g. a Kaggle anemia dataset)."

    def add_arguments(self, parser):
        parser.add_argument("csv_path")
        parser.add_argument("--limit", type=int, default=30)
        parser.add_argument("--render", choices=["text", "pdf", "png"], default="text")
        parser.add_argument("--no-ranges", action="store_true",
                            help="Do not print reference ranges (tests the fallback-range path; no flag agreement)")
        parser.add_argument("--out", default="eval_results.csv")

    def handle(self, *args, **o):
        path = Path(o["csv_path"])
        if not path.exists():
            raise CommandError(f"File not found: {path}")
        rows = read_rows(path)
        header_idx = next((i for i, r in enumerate(rows) if len(map_header(r)["cols"]) >= 3), None)
        if header_idx is None:
            raise CommandError("No header row with at least 3 recognisable CBC columns (HGB, RBC, MCV, WBC/TLC, PLT...).")
        m = map_header(rows[header_idx])
        self.stdout.write(f"Header found on line {header_idx + 1}. Column mapping:")
        for key, i in m["cols"].items():
            self.stdout.write(f"  '{rows[header_idx][i]}' -> {key}")
        for name, i in (("sex", m["sex"]), ("age", m["age"]), ("label", m["label"])):
            if i is not None:
                self.stdout.write(f"  '{rows[header_idx][i]}' -> {name}")
        ranges = load_reference_ranges()["tests"]
        ranged = not o["no_ranges"]

        used = skipped = 0
        ext_ok = ext_total = ext_missing = ext_wrong = 0
        flag_ok = flag_total = 0
        flag_dist, crit_rows = Counter(), 0
        cross = defaultdict(Counter)
        problems, out_rows, t0 = [], [], time.time()

        for r in rows[header_idx + 1:]:
            if used >= o["limit"]:
                break
            vals = {k: to_float(r[i]) for k, i in m["cols"].items() if i < len(r)}
            vals = {k: v for k, v in vals.items() if v is not None}
            if len(vals) < 3:
                skipped += 1
                continue
            sex = parse_sex(r[m["sex"]]) if m["sex"] is not None and m["sex"] < len(r) else None
            age = to_float(r[m["age"]]) if m["age"] is not None and m["age"] < len(r) else None
            label = r[m["label"]].strip() if m["label"] is not None and m["label"] < len(r) else ""

            lines = ["CITY DIAGNOSTIC LAB - COMPLETE BLOOD COUNT",
                     f"Patient: Row {used + 1}   " + (f"Age: {int(age)}   " if age else "")
                     + (f"Sex: {sex.capitalize()}" if sex else ""),
                     "Test            Result    Unit        Reference Range"]
            truth = {}
            for key, v in vals.items():
                if key in ("wbc", "platelets"):
                    per_cumm = v >= (500 if key == "wbc" else 2000)
                    unit, scale = ("/cumm", 1000) if per_cumm else ("x10^3/uL", 1)
                else:
                    unit, scale = BASE_UNITS[key], 1
                lo = hi = None
                if ranged:
                    ent = ranges[key]
                    pair = ent[sex] if sex else [min(ent["male"][0], ent["female"][0]), max(ent["male"][1], ent["female"][1])]
                    lo, hi = pair[0] * scale, pair[1] * scale
                rng = f"{fmt(lo)} - {fmt(hi)}" if ranged else ""
                lines.append(f"{PRINT_NAMES[key]}   {fmt(v)}   {unit}   {rng}".rstrip())
                truth[key] = (v, lo, hi)
            text = "\n".join(lines)

            try:
                if o["render"] == "text":
                    ex = extract_from_text(text)
                elif o["render"] == "pdf":
                    ex = extract_from_bytes("r.pdf", render_pdf(text))
                else:
                    ex = extract_from_bytes("r.png", render_png(text))
                res = run_analysis(ex, "en")
            except Exception as exc:
                problems.append(f"row {used + 1}: crashed: {exc}")
                used += 1
                continue

            got = {canonical_key(t.test): t.value for t in ex.tests if canonical_key(t.test)}
            row_ok = True
            for key, (v, lo, hi) in truth.items():
                ext_total += 1
                if key not in got:
                    ext_missing += 1
                    row_ok = False
                    problems.append(f"row {used + 1}: {key} not found (expected {fmt(v)})")
                elif abs(got[key] - v) > 0.005 * max(abs(v), 1):
                    ext_wrong += 1
                    row_ok = False
                    problems.append(f"row {used + 1}: {key} read as {fmt(got[key])}, expected {fmt(v)}")
                else:
                    ext_ok += 1
            by_key = {x["key"]: x for x in res["results"]}
            hb_flag = ""
            for key, (v, lo, hi) in truth.items():
                x = by_key.get(key)
                if not x:
                    continue
                shown = {"low": "Low", "high": "High"}.get(x["direction"], "Normal") if x["flag"] != "Not assessed" else "Not assessed"
                flag_dist[x["flag"]] += 1
                if key == "hemoglobin":
                    hb_flag = x["flag"]
                if ranged and key in got and abs(got[key] - v) <= 0.005 * max(abs(v), 1):
                    expected = "Low" if v < lo else "High" if v > hi else "Normal"
                    flag_total += 1
                    if shown == expected:
                        flag_ok += 1
                    else:
                        problems.append(f"row {used + 1}: {key} flag {shown}, expected {expected}")
            crit_rows += 1 if res["alerts"] else 0
            if label:
                cross[label][hb_flag or "n/a"] += 1
            out_rows.append({"row": used + 1, "label": label, "extraction_ok": row_ok, "hb_flag": hb_flag,
                             "alerts": len(res["alerts"]), "explanation_source": res["explanation_source"],
                             "verified": res["verification"]["passed"], "source": ex.source})
            used += 1

        secs = time.time() - t0
        w = self.stdout.write
        w(f"\nRows analysed: {used} (skipped {skipped} non-data rows), render={o['render']}, {secs:.1f}s")
        if ext_total:
            w(f"Extraction: {ext_ok}/{ext_total} values read back correctly ({100 * ext_ok / ext_total:.1f}%), "
              f"missing {ext_missing}, wrong {ext_wrong}")
        if ranged and flag_total:
            w(f"Flags: {flag_ok}/{flag_total} agree with a direct comparison to the printed range "
              f"({100 * flag_ok / flag_total:.1f}%)")
        elif not ranged:
            w("Flags: skipped (--no-ranges: fallback ranges are the same table the rules use, so it would be circular)")
        w("Flag counts: " + ", ".join(f"{k}={v}" for k, v in sorted(flag_dist.items())))
        w(f"Rows with a critical-value alert: {crit_rows} (limits are placeholders)")
        verified = sum(1 for x in out_rows if x["verified"])
        w(f"Explanations passing verification: {verified}/{len(out_rows)}")
        if cross:
            w("\nDataset label vs our hemoglobin flag (sanity check only, not our accuracy):")
            flags = ["Low", "Normal", "High", "Critical", "n/a"]
            w("  " + f"{'label':28}" + "".join(f"{f:>9}" for f in flags))
            for lab, cnt in sorted(cross.items()):
                w("  " + f"{lab[:27]:28}" + "".join(f"{cnt.get(f, 0):>9}" for f in flags))
        if problems:
            w(f"\n{len(problems)} problem(s), first 10:")
            for p in problems[:10]:
                w("  - " + p)
        with open(o["out"], "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(out_rows[0]) if out_rows else ["row"])
            wr.writeheader()
            wr.writerows(out_rows)
        w(f"\nPer-row results written to {o['out']}")
