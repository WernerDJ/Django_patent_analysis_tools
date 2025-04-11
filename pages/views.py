from django.shortcuts import render
from django.views.generic import TemplateView
from django.views.generic import FormView
from django.core.cache import cache
from django.http import JsonResponse
from .forms import ExcelUploadForm
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import functions from your analytic module
from pages.analytic_functions import (
    extract_main_ipc, 
    filter_data_by_ipc_and_year, 
    plot_top_10_countries, 
    plot_top_10_ipcs
)

class HomePageView(TemplateView):
    template_name = "home.html"

class AboutPageView(TemplateView):
    template_name = "about.html"


class AnalizarDatosView(FormView):
    template_name = "pages/analisis.html"
    form_class = ExcelUploadForm
    # You might define a success_url or handle redirection differently
    extra_context = {}

    def form_valid(self, form):
        excel_file = form.cleaned_data.get('excel_file')
        ipc_input = form.cleaned_data.get('ipc_groups', "")
        # Convert comma-separated IPC groups into a list
        ipc_list = [ipc.strip() for ipc in ipc_input.split(",") if ipc.strip()]
        start_year = form.cleaned_data.get('start_year')
        end_year = form.cleaned_data.get('end_year')
        time_range = [start_year, end_year]

        try:
            # Read the Excel file into a DataFrame
            df = pd.read_excel(excel_file, skiprows=5)
            
            # Apply MainIPC extraction and Publication Year extraction
            df['MainIPC'] = df['I P C'].apply(extract_main_ipc)
            df['PubYear'] = pd.to_datetime(
                df['Publication Date'], format='%d.%m.%Y', errors='coerce'
            ).dt.year

            # Filter the DataFrame based on IPCs and Time Range
            filtered_data = filter_data_by_ipc_and_year(df, ipc_list, time_range)
            
            # Generate and save plots
            # Ensure your static images directory exists; adjust path as needed.
            base_path = "static/images/"
            os.makedirs(base_path, exist_ok=True)
            
            # Plot top 10 countries
            plt_countries = plot_top_10_countries(filtered_data)
            countries_img_path = os.path.join(base_path, "top_countries.png")
            plt_countries.savefig(countries_img_path)
            plt.close()  # Close the figure

            # Plot top 10 IPCs
            plt_ipcs = plot_top_10_ipcs(filtered_data)
            ipcs_img_path = os.path.join(base_path, "top_ipcs.png")
            plt_ipcs.savefig(ipcs_img_path)
            plt.close()
            
            extra_context = {
                'analysis1': "<p>Analysis complete. See the graphs below.</p>",
                'top_countries_img': countries_img_path,
                'top_ipcs_img': ipcs_img_path,
            }
        except Exception as e:
            extra_context = {'error': f"An error occurred during file processing: {e}"}

        context = self.get_context_data(form=form, **extra_context)
        return self.render_to_response(context)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if hasattr(self, 'extra_context'):
            context.update(self.extra_context)
        return context
