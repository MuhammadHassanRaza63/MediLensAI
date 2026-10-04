from django.urls import path

from . import views

urlpatterns = [
    path("", views.ReportListCreate.as_view()),
    path("<uuid:pk>/", views.ReportDetail.as_view()),
    path("<uuid:pk>/confirm/", views.ReportConfirm.as_view()),
]
