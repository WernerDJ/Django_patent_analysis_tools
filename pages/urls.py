from django.urls import path
from .views import HomePageView, AboutPageView, AnalizarDatosView, ApplicInventNetworkView, CountriesView # , filter_data

urlpatterns = [
    path("", HomePageView.as_view(), name="home"),
    path("countries", CountriesView.as_view(), name="countries"),
    path("analisis/", AnalizarDatosView.as_view(), name="analisis"),
    path("network/", ApplicInventNetworkView.as_view(), name = "network"),
    path("about/", AboutPageView.as_view(), name="about"),
]
