from django.conf import settings
from django.core.files.base import ContentFile
from pydantic import ValidationError
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from core.safety import DISCLAIMERS

from .models import ReportUpload
from .services.extract import detect_kind, extract_from_bytes, extract_from_text
from .services.pipeline import extraction_needs_confirmation, run_analysis
from .services.schemas import ExtractedTest, ExtractionResult, Patient

MAX_BYTES = 15 * 1024 * 1024


def _payload(rep: ReportUpload) -> dict:
    data = {"id": str(rep.id), "status": rep.status, "created_at": rep.created_at,
            "language": rep.language, "extraction": rep.extraction}
    if rep.reasons:
        data["needs_confirmation_because"] = rep.reasons
    if rep.analysis:
        data["analysis"] = rep.analysis
    return data


def _parse_age(v):
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _parse_sex(v):
    v = (v or "").strip().lower()
    return v if v in ("male", "female") else None


class ReportListCreate(APIView):
    """POST a CBC report as `file` (PDF/PNG/JPG) or pasted `text`."""

    def post(self, request):
        language = request.data.get("language", "en")
        sex, age = _parse_sex(request.data.get("sex")), _parse_age(request.data.get("age"))
        upload = request.FILES.get("file")
        text = request.data.get("text")
        if not upload and not text:
            return Response({"error": "Send a `file` (PDF/PNG/JPG) or `text`."}, status=400)

        rep = ReportUpload(language=language)
        try:
            if upload:
                if upload.size > MAX_BYTES:
                    return Response({"error": "File is larger than 15 MB."}, status=413)
                data = upload.read()
                rep.original_name = upload.name[:255]
                rep.kind = detect_kind(upload.name, data)
                if rep.kind == "unknown":
                    return Response({"error": "Unsupported file type. Upload a PDF, PNG or JPG."}, status=415)
                extraction = extract_from_bytes(upload.name, data)
                if not settings.MEDILENS_DELETE_UPLOADS_AFTER_PROCESSING:
                    rep.file.save(upload.name, ContentFile(data), save=False)
                else:
                    rep.file_deleted = True
            else:
                rep.kind = "text"
                extraction = extract_from_text(str(text))
        except Exception as exc:  # corrupt PDF/image etc.
            rep.status, rep.reasons = ReportUpload.Status.FAILED, [f"Could not read the file: {exc}"]
            rep.save()
            return Response(_payload(rep), status=422)

        if sex or age:
            extraction.patient = Patient(sex=sex or extraction.patient.sex, age=age or extraction.patient.age)
        rep.extraction = extraction.model_dump()
        reasons = extraction_needs_confirmation(extraction)
        if reasons:
            rep.status, rep.reasons = ReportUpload.Status.NEEDS_CONFIRMATION, reasons
        else:
            rep.analysis = run_analysis(extraction, language)
            rep.status = ReportUpload.Status.COMPLETED
        rep.save()
        return Response(_payload(rep), status=201)


class ReportDetail(APIView):
    def get(self, request, pk):
        try:
            rep = ReportUpload.objects.get(pk=pk)
        except ReportUpload.DoesNotExist:
            return Response({"error": "Not found."}, status=404)
        return Response(_payload(rep))

    def delete(self, request, pk):
        """Privacy: remove the stored file and all extracted data."""
        try:
            rep = ReportUpload.objects.get(pk=pk)
        except ReportUpload.DoesNotExist:
            return Response({"error": "Not found."}, status=404)
        if rep.file:
            rep.file.delete(save=False)
        rep.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ReportConfirm(APIView):
    """User checks/corrects the extracted values, then the analysis runs on exactly those values."""

    def post(self, request, pk):
        try:
            rep = ReportUpload.objects.get(pk=pk)
        except ReportUpload.DoesNotExist:
            return Response({"error": "Not found."}, status=404)
        tests_in = request.data.get("tests")
        if not isinstance(tests_in, list) or not tests_in:
            return Response({"error": "Send `tests`: a list of {test, value, unit, ref_range}."}, status=400)
        try:
            tests = [ExtractedTest(**{**t, "confidence": 1.0}) for t in tests_in]
        except (ValidationError, TypeError) as exc:
            return Response({"error": "Invalid test entry.", "detail": str(exc)[:500]}, status=400)
        old = rep.extraction.get("patient") or {}
        sex = _parse_sex(request.data.get("sex")) or old.get("sex")
        age = _parse_age(request.data.get("age")) or old.get("age")
        language = request.data.get("language", rep.language)
        extraction = ExtractionResult(tests=tests, patient=Patient(sex=sex, age=age), source="user")
        rep.language, rep.extraction, rep.reasons = language, extraction.model_dump(), []
        rep.analysis = run_analysis(extraction, language)
        rep.status = ReportUpload.Status.COMPLETED
        rep.save()
        return Response(_payload(rep))
