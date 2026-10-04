"""Pydantic schemas shared across the report pipeline."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ExtractedTest(BaseModel):
    test: str = Field(min_length=1)
    value: float
    unit: Optional[str] = None
    ref_range: Optional[str] = None      # exactly as printed, e.g. "12.0 - 15.5"
    page: int = 1
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    flag_raw: Optional[str] = None        # H / L printed by the lab, if any

    @field_validator("test")
    @classmethod
    def _strip(cls, v):
        return v.strip()


class Patient(BaseModel):
    sex: Optional[Literal["male", "female"]] = None
    age: Optional[int] = Field(default=None, ge=0, le=130)


class ExtractionResult(BaseModel):
    tests: list[ExtractedTest] = []
    patient: Patient = Patient()
    source: str = "text"                  # text | pdf_text | ocr | vision_llm | ocr+vision_llm | user
    warnings: list[str] = []
