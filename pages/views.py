from django.shortcuts import render
from django.views.generic import TemplateView, FormView
from .forms import ExcelUploadForm, SimpleExcelUploadForm
import os
from .analytic_functions import Patent_Analysis, Patent_Network
import uuid

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

            base_path = "static/images/temp/"
            os.makedirs(base_path, exist_ok=True)

            # Plot priority patent filling frequencies timeline
            plt_priority_years = analyzer.plot_priority_years_bar(priority_df)
            priority_years_img_path = os.path.join(base_path, f"frequency_priority_years_{filename_suffix}.png")
            plt_priority_years.savefig(priority_years_img_path)

            # Plot top priority countries
            plt_priority_countries = analyzer.plot_priority_countries_bar(priority_df)
            priority_countries_img_path = os.path.join(base_path, f"top_priority_countries_{filename_suffix}.png")
            plt_priority_countries.savefig(priority_countries_img_path)

            # Plot top publiction countries
            plt_countries = analyzer.plot_top_10_countries()
            countries_img_path = os.path.join(base_path, f"top_countries_{filename_suffix}.png")
            plt_countries.savefig(countries_img_path)
            plt_countries.close()

            # Origin and Destination Countries
            plt_origin_destcountr = analyzer.analyze_patent_flow(top_n=10)
            origin_destcountr_img_path = os.path.join(base_path, f"origin_destcountr_{filename_suffix}.png")
            plt_origin_destcountr.savefig(origin_destcountr_img_path)
            plt_origin_destcountr.close()

            extra_context = {
                    'analysis1': "<p>Analysis complete. See the graphs below.</p>",
                    'priority_years_img': priority_years_img_path,
                    'priority_countries_img': priority_countries_img_path,
                    'top_countries_img': countries_img_path,
                    'origin_destcountr_img':origin_destcountr_img_path,
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
            base_path = "static/images/temp/"
            os.makedirs(base_path, exist_ok=True)

            # Plot wordcloud nouns
            plt_wcld_nouns = analyzer.generate_wordclouds_by_pos(pospeech = 'Nouns')
            wcld_nouns_img_path = os.path.join(base_path, f'wcld_nouns_{filename_suffix}.png')
            plt_wcld_nouns.savefig(wcld_nouns_img_path)
            plt_wcld_nouns.close()

            # Plot wordcloud verbs
            plt_wcld_verbs = analyzer.generate_wordclouds_by_pos(pospeech = 'Verbs')
            wcld_verbs_img_path = os.path.join(base_path, f'wcld_verbs_{filename_suffix}.png')
            plt_wcld_verbs.savefig(wcld_verbs_img_path)
            plt_wcld_verbs.close()

            # Plot wordcloud adjectives
            plt_wcld_adjectives = analyzer.generate_wordclouds_by_pos(pospeech = 'Adjectives')
            wcld_adjectives_img_path = os.path.join(base_path, f'wcld_adjectives_{filename_suffix}.png')
            plt_wcld_adjectives.savefig(wcld_adjectives_img_path)
            plt_wcld_adjectives.close()            

            # Plot top IPCs
            plt_ipcs = analyzer.plot_top_ipcs()
            ipcs_img_path = os.path.join(base_path, f"top_ipcs_{filename_suffix}.png")
            plt_ipcs.savefig(ipcs_img_path)
            plt_ipcs.close()

            # Show the boring table with IPC Groups and their definitions
            plt_defs = analyzer.get_top_ipcs_with_titles()
            defs_img_path = os.path.join(base_path, f"top_ipcs_defs_{filename_suffix}.png")
            plt_defs.savefig(defs_img_path)
            plt_defs.close()

            # Plot parallel coordinates
            plt_parallel = analyzer.plot_parallel_coordinates(top_n=5, year_range=range(start_year, end_year + 1))
            parallel_img_path = os.path.join(base_path, f"parallel_coordinates_{filename_suffix}.png")
            plt_parallel.savefig(parallel_img_path)
            

            extra_context = {
                'analysis1': "<p>Analysis complete. See the graphs below.</p>",
                'top_ipcs_img': ipcs_img_path,
                'top_ipcs_defs_img':defs_img_path,
                'parallel_img': parallel_img_path,
                'wcld_nouns_img':wcld_nouns_img_path,
                'wcld_verbs_img':wcld_verbs_img_path,
                'wcld_adjectives_img':wcld_adjectives_img_path,
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
            base_path = os.path.join("static", "images", "temp")
            os.makedirs(base_path, exist_ok=True)
            network_img_path = os.path.join(base_path, f"app_inv_network_{filename_suffix}.png")
            print(f"Image will be saved to: {os.path.abspath(network_img_path)}")
            
            # Process the data and generate the network visualization
            print("Filtering data...")
            analyzer.filter_data()
            print("Building graph...")
            analyzer.build_graph()
            print("Plotting network...")

            # Instead of getting base64, save the figure directly
            plt_network = analyzer.generate_network_image(top_n=20)  # Assuming this returns a matplotlib figure
            plt_network.savefig(network_img_path)
            extra_context['network_img'] = network_img_path

        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            extra_context = {'error': f"An error occurred during file processing: {e}"}
        
        print(f"Final extra_context: {extra_context}")
        context = self.get_context_data(form=form, **extra_context)
        return self.render_to_response(context)
