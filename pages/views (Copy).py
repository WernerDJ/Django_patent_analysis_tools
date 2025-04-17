from django.shortcuts import render
from django.views.generic import TemplateView, FormView
from .forms import ExcelUploadForm
import os
from .analytic_functions import Patent_Analysis  # Use the full class

class HomePageView(TemplateView):
    template_name = "home.html"

class AboutPageView(TemplateView):
    template_name = "about.html"

class AnalizarDatosView(FormView):
    template_name = "pages/analisis.html"
    form_class = ExcelUploadForm

    def form_valid(self, form):
        excel_file = form.cleaned_data.get('excel_file')
        ipc_input = form.cleaned_data.get('ipc_groups', "")
        ipc_list = [ipc.strip() for ipc in ipc_input.split(",") if ipc.strip()]
        start_year = form.cleaned_data.get('start_year')
        end_year = form.cleaned_data.get('end_year')
        time_range = [start_year, end_year]

        try:
            # Initialize the Patent_Analysis object
            analyzer = Patent_Analysis(excel_file)

            # Filter the data
            analyzer.filter_by_ipc_and_year(ipc_list, time_range)

            # Define image save path
            base_path = "static/images/"
            os.makedirs(base_path, exist_ok=True)

            # Plot top countries
            plt_countries = analyzer.plot_top_10_countries()
            countries_img_path = os.path.join(base_path, "top_countries.png")
            plt_countries.savefig(countries_img_path)
            plt_countries.close()

            # Plot top IPCs
            plt_ipcs = analyzer.plot_top_ipcs()
            ipcs_img_path = os.path.join(base_path, "top_ipcs.png")
            plt_ipcs.savefig(ipcs_img_path)
            plt_ipcs.close()

            # Plot parallel coordinates
            plt_parallel = analyzer.plot_parallel_coordinates(top_n=5, year_range=range(start_year, end_year + 1))
            parallel_img_path = os.path.join(base_path, "parallel_coordinates.png")
            plt_parallel.savefig(parallel_img_path)
            plt_parallel.close()

            extra_context = {
                'analysis1': "<p>Analysis complete. See the graphs below.</p>",
                'top_countries_img': countries_img_path,
                'top_ipcs_img': ipcs_img_path,
                'parallel_img': parallel_img_path,
            }
        except Exception as e:
            extra_context = {'error': f"An error occurred during file processing: {e}"}

        context = self.get_context_data(form=form, **extra_context)
        return self.render_to_response(context)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context
