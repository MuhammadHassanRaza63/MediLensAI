from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path

from core.llm import get_llm


def health(request):
    return JsonResponse({"status": "ok", "llm_available": get_llm().available})


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/health/", health),
    path("api/reports/", include("reports.urls")),
    path("api/medicines/", include("medicines.urls")),
]
