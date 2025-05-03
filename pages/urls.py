from django.urls import path
from .views import HomePageView, AboutPageView, AnalizarDatosView, IPC_ApplicantsView, ApplicInventNetworkView, CountriesView # , filter_data

urlpatterns = [
    path("", HomePageView.as_view(), name="home"),
    path("countries", CountriesView.as_view(), name="countries"),
    path("analisis/", AnalizarDatosView.as_view(), name="analisis"),
    path("IPC_Applicants/", IPC_ApplicantsView.as_view(), name="IPC_Applicants"),
    path("network/", ApplicInventNetworkView.as_view(), name = "network"),
    path("about/", AboutPageView.as_view(), name="about"),
]
