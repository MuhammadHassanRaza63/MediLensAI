"""Cheap image-quality gate so we ask for a better photo instead of guessing."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

BLUR_MIN = 40.0         # variance of Laplacian on a <=1000px grayscale copy
DARK_MAX_MEAN = 45
BRIGHT_MIN_MEAN = 215
GLARE_FRACTION = 0.12
MIN_SIDE = 400


def _laplacian_var(a: np.ndarray) -> float:
    lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
    return float(lap.var())


def assess_image(data: bytes) -> dict:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    g = ImageOps.grayscale(img)
    if max(g.size) > 1000:
        s = 1000 / max(g.size)
        g = g.resize((int(g.width * s), int(g.height * s)))
    a = np.asarray(g, dtype=np.float32)
    blur = _laplacian_var(a) if min(a.shape) > 3 else 0.0
    mean = float(a.mean())
    glare = float((a >= 250).mean())
    issues, warnings = [], []     # issues block a confident answer; warnings are advisory
    if min(w, h) < MIN_SIDE:
        issues.append("The photo is too small. Move closer or use a higher resolution.")
    if blur < BLUR_MIN:
        issues.append("The photo looks blurry. Hold the phone steady and let it focus.")
    if mean < DARK_MAX_MEAN:
        issues.append("The photo is too dark. Use better light.")
    if mean > BRIGHT_MIN_MEAN or glare > GLARE_FRACTION:
        # A white pack on a white table looks the same as glare, so this only warns.
        warnings.append("There may be glare or the photo may be washed out. Tilt the pack to avoid reflections.")
    return {"ok": not issues, "issues": issues, "warnings": warnings, "blur_score": round(blur, 1),
            "brightness": round(mean, 1), "glare_fraction": round(glare, 3), "size": [w, h]}
