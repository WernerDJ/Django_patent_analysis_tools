from django.conf import settings
from django.shortcuts import render
from django.views.generic import TemplateView, FormView
from .forms import ExcelUploadForm, SimpleExcelUploadForm
import os
from .analytic_functions import Patent_Analysis, Patent_Network
import uuid

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

# Generate unique ID for this session
filename_suffix = uuid.uuid4().hex[:8]  

class HomePageView(TemplateView):
    template_name = "home.html"

class AboutPageView(TemplateView):
    template_name = "about.html"


class CountriesView(FormView):
    template_name = "pages/countries.html"
    form_class = ExcelUploadForm

    def form_valid(self, form):
        excel_file = form.cleaned_data.get('excel_file')
        ipc_input = form.cleaned_data.get('ipc_groups', "")
        ipc_list = [ipc.strip() for ipc in ipc_input.split(",") if ipc.strip()]
        start_year = form.cleaned_data.get('start_year')
        end_year = form.cleaned_data.get('end_year')
        time_range = [start_year, end_year]

        try:
            analyzer = Patent_Analysis(excel_file)
            analyzer.filter_by_ipc_and_year(ipc_list, time_range)
            # Call prepare_priority_data from here
            priority_df = analyzer.prepare_priority_data()

            base_path = os.path.join(settings.MEDIA_ROOT, "images")
            os.makedirs(base_path, exist_ok=True)

            # Plot priority patent filling frequencies timeline
            plt_priority_years = analyzer.plot_priority_years_bar(priority_df)
            filename = f"frequency_priority_years_{filename_suffix}.png"
            priority_years_img_path = os.path.join(base_path, filename)
            plt_priority_years.savefig(priority_years_img_path)
            priority_years_img_url = settings.MEDIA_URL + f"images/{filename}"

            '''
            priority_years_img_path = os.path.join(base_path, f"frequency_priority_years_{filename_suffix}.png")
            plt_priority_years.savefig(priority_years_img_path)
            '''

            # Plot top priority countries
            plt_priority_countries = analyzer.plot_priority_countries_bar(priority_df)
            filename = f"frequency_priority_countries_{filename_suffix}.png"
            priority_countries_img_path = os.path.join(base_path, filename)
            plt_priority_countries.savefig(priority_countries_img_path)
            priority_countries_img_url = settings.MEDIA_URL + f"images/{filename}"

            # Plot top publiction countries
            plt_countries = analyzer.plot_top_10_countries()
            filename = f"top_countries_{filename_suffix}.png"
            countries_img_path  = os.path.join(base_path, filename)
            plt_countries.savefig(countries_img_path)
            countries_img_url = settings.MEDIA_URL + f"images/{filename}" #it might need to be renamed to top_countries
 
            # Origin and Destination Countries
            plt_origin_destcountr = analyzer.analyze_patent_flow(top_n=10)
            filename = f"origin_destcountr_{filename_suffix}.png"
            origin_destcountr_img_path  = os.path.join(base_path, filename)
            plt_origin_destcountr.savefig(origin_destcountr_img_path)
            origin_destcountr_img_url = settings.MEDIA_URL + f"images/{filename}" #it might need to be renamed to top_countries

            extra_context = {
                    'analysis1': "<p>Analysis complete. See the graphs below.</p>",
                    'priority_years_img_url': priority_years_img_url,
                    'priority_countries_img_url': priority_countries_img_url,
                    'countries_img_url': countries_img_url,
                    'origin_destcountr_img_url':origin_destcountr_img_url,
                    }

        except Exception as e:
            extra_context = {'error': f"An error occurred during file processing: {e}"}

        context = self.get_context_data(form=form, **extra_context)
        return self.render_to_response(context)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


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
            base_path =  os.path.join(settings.MEDIA_ROOT, "images")
            os.makedirs(base_path, exist_ok=True)
            
            # Plot wordcloud nouns
            plt_wcld_nouns = analyzer.generate_wordclouds_by_pos(pospeech = 'Nouns')
            filename = f"wcld_nouns_{filename_suffix}.png"
            wcld_nouns_img_path  = os.path.join(base_path, filename)
            plt_wcld_nouns.savefig(wcld_nouns_img_path)
            wcld_nouns_img_url = settings.MEDIA_URL + f"images/{filename}"

            # Plot wordcloud verbs
            plt_wcld_verbs = analyzer.generate_wordclouds_by_pos(pospeech = 'Verbs')
            filename = f"wcld_verbs_{filename_suffix}.png"
            wcld_verbs_img_path  = os.path.join(base_path, filename)
            plt_wcld_verbs.savefig(wcld_verbs_img_path)
            wcld_verbs_img_url = settings.MEDIA_URL + f"images/{filename}"

            # Plot wordcloud adjectives
            plt_wcld_adjectives = analyzer.generate_wordclouds_by_pos(pospeech = 'Adjectives')
            filename = f"wcld_adjectives_{filename_suffix}.png"
            wcld_adjectives_img_path  = os.path.join(base_path, filename)
            plt_wcld_adjectives.savefig(wcld_adjectives_img_path)
            wcld_adjectives_img_url = settings.MEDIA_URL + f"images/{filename}"     
    
            # Plot top IPCs
            plt_ipcs = analyzer.plot_top_ipcs()
            filename = f"top_ipcs_{filename_suffix}.png"
            ipcs_img_path  = os.path.join(base_path, filename)
            plt_ipcs.savefig(ipcs_img_path)
            ipcs_img_url = settings.MEDIA_URL + f"images/{filename}" 

            # Show the boring table with IPC Groups and their definitions
            plt_defs = analyzer.get_top_ipcs_with_titles()
            filename = f"top_ipcs_defs_{filename_suffix}.png"
            defs_img_path  = os.path.join(base_path, filename)
            plt_defs.savefig(ipcs_img_path)
            defs_img_url = settings.MEDIA_URL + f"images/{filename}" 

            # Plot parallel coordinates
            plt_parallel = analyzer.plot_parallel_coordinates(top_n=5, year_range=range(start_year, end_year + 1))
            filename = f"parallel_coordinates_{filename_suffix}.png"
            parallel_img_path  = os.path.join(base_path, filename)
            plt_parallel.savefig(parallel_img_path)
            parallel_img_url = settings.MEDIA_URL + f"images/{filename}" 
            

            extra_context = {
                'analysis1': "<p>Analysis complete. See the graphs below.</p>",
                'top_ipcs_img': ipcs_img_url,
                'top_ipcs_defs_img':defs_img_url,
                'parallel_img': parallel_img_url, 
                'wcld_nouns_img':wcld_nouns_img_url,
                'wcld_verbs_img':wcld_verbs_img_url,
                'wcld_adjectives_img':wcld_adjectives_img_url,
                } 
            

        except Exception as e:
            extra_context = {'error': f"An error occurred during file processing: {e}"}

        context = self.get_context_data(form=form, **extra_context)
        return self.render_to_response(context)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

class ApplicInventNetworkView(FormView):
    template_name = "pages/network.html"
    form_class = SimpleExcelUploadForm
    
    def form_valid(self, form):
        excel_file = form.cleaned_data.get('excel_file')
        print(f"Excel file received: {excel_file}")
        
        extra_context = {}
        
        try:
            print("Initializing PatentNetwork...")
            analyzer = Patent_Network(excel_file)

            # Define image save path
            base_path = os.path.join(settings.MEDIA_ROOT, "images")
            os.makedirs(base_path, exist_ok=True)

            '''
            network_img_path = os.path.join(base_path, f"app_inv_network_{filename_suffix}.png")
            print(f"Image will be saved to: {os.path.abspath(network_img_path)}")
            '''

            # Process the data and generate the network visualization
            print("Filtering data...")
            analyzer.filter_data()
            print("Building graph...")
            analyzer.build_graph()
            print("Plotting network...")

            # Instead of getting base64, save the figure directly
            plt_network = analyzer.generate_network_image(top_n=20)  # Assuming this returns a matplotlib figure
            filename = f"network_{filename_suffix}.png"
            network_img_path  = os.path.join(base_path, filename)
            plt_network.savefig(network_img_path)
            network_img_url = settings.MEDIA_URL + f"images/{filename}" 
            
            extra_context = {
                'analysis1': "<p>Analysis complete. See the graphs below.</p>",
                'network_img_url': network_img_url,
                }
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            extra_context = {'error': f"An error occurred during file processing: {e}"}
        
        print(f"Final extra_context: {extra_context}")
        context = self.get_context_data(form=form, **extra_context)
        return self.render_to_response(context)
