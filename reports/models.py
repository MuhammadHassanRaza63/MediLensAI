import uuid

from django.db import models


class ReportUpload(models.Model):
    class Status(models.TextChoices):
        NEEDS_CONFIRMATION = "needs_confirmation"
        COMPLETED = "completed"
        FAILED = "failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    original_name = models.CharField(max_length=255, blank=True)
    kind = models.CharField(max_length=16, blank=True)
    file = models.FileField(upload_to="reports/", null=True, blank=True)
    file_deleted = models.BooleanField(default=False)
    language = models.CharField(max_length=16, default="en")
    status = models.CharField(max_length=24, choices=Status.choices, default=Status.FAILED)
    extraction = models.JSONField(default=dict, blank=True)
    analysis = models.JSONField(default=dict, blank=True)
    reasons = models.JSONField(default=list, blank=True)

    def __str__(self):
        return f"Report {self.id} ({self.status})"
