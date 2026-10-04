"""Render report text as a PDF or PNG (used by the evaluation commands)."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
]


def get_font(size: int):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render_png(text: str, width: int = 1500, size: int = 30) -> bytes:
    lines = text.strip().splitlines()
    img = Image.new("RGB", (width, 60 + int(size * 1.7) * len(lines)), "white")
    d = ImageDraw.Draw(img)
    font = get_font(size)
    for i, line in enumerate(lines):
        d.text((40, 30 + int(size * 1.7) * i), line, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def render_pdf(text: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in text.strip().splitlines():
        c.drawString(40, y, line)
        y -= 16
    c.save()
    return buf.getvalue()
