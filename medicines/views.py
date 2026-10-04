from rest_framework.response import Response
from rest_framework.views import APIView

from .services import agent

MAX_BYTES = 10 * 1024 * 1024
ALLOWED = ("image/jpeg", "image/png", "image/webp")


class Identify(APIView):
    """POST a photo of the medicine pack as `image` (optionally `ocr_text`, `language`)."""

    def post(self, request):
        image = request.FILES.get("image")
        ocr_text = request.data.get("ocr_text")
        language = request.data.get("language", "en")
        if not image and not ocr_text:
            return Response({"error": "Send an `image` (JPG/PNG) of the pack."}, status=400)
        data, media = None, "image/jpeg"
        if image:
            if image.size > MAX_BYTES:
                return Response({"error": "Image is larger than 10 MB."}, status=413)
            media = image.content_type or "image/jpeg"
            if media not in ALLOWED:
                return Response({"error": "Use a JPG, PNG or WEBP image."}, status=415)
            data = image.read()          # processed in memory; never written to disk
        try:
            return Response(agent.identify(data, media, ocr_text, language))
        except Exception as exc:
            return Response({"status": "failed", "error": f"Could not process the image: {exc}"}, status=422)


class Confirm(APIView):
    def post(self, request):
        brand = request.data.get("brand")
        if not brand:
            return Response({"error": "Send `brand` (and optionally `strength`, `form`)."}, status=400)
        out = agent.confirm(brand, request.data.get("strength"), request.data.get("form"),
                            request.data.get("language", "en"))
        if out is None:
            return Response({"error": "No such medicine in the table."}, status=404)
        return Response(out)
