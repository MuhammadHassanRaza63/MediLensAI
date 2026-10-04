from django.urls import path

from . import views

urlpatterns = [
    path("identify/", views.Identify.as_view()),
    path("confirm/", views.Confirm.as_view()),
]
