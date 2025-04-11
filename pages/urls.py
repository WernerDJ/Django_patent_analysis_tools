from django.urls import path
from .views import HomePageView, AboutPageView, AnalizarDatosView  # , filter_data

urlpatterns = [
    path("", HomePageView.as_view(), name="home"),
    path("analisis/", AnalizarDatosView.as_view(), name="analisis"),
    path("about/", AboutPageView.as_view(), name="about"),
]
