from django.conf import settings
from django.shortcuts import render
from django.views.generic import TemplateView, FormView
from .forms import ExcelUploadForm, SimpleExcelUploadForm, ReducedExcelUploadForm
import os
from .analytic_functions import Patent_Analysis, Patent_Network
import uuid

import matplotlib
from cloudinary.uploader import upload
from cloudinary.utils import cloudinary_url

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


            # Plot priority patent filling frequencies timeline
            plt_priority_years = analyzer.plot_priority_years_bar(priority_df)
            filename_priority_years = f"frequency_priority_years_{filename_suffix}.png"
            temp_priority_years_path = f"/tmp/{filename_priority_years}"
            plt_priority_years.savefig(temp_priority_years_path) # Save locally first
            uploaded_priority_years = upload(temp_priority_years_path)
            priority_years_img_url = uploaded_priority_years['secure_url']
            os.remove(temp_priority_years_path)

            '''
            priority_years_img_path = os.path.join(base_path, f"frequency_priority_years_{filename_suffix}.png")
            plt_priority_years.savefig(priority_years_img_path)
            '''

            # Plot top priority countries
            plt_priority_countries = analyzer.plot_priority_countries_bar(priority_df)
            filename_priority_countries = f"frequency_priority_countries_{filename_suffix}.png"
            temp_priority_countries_path = f"/tmp/{filename_priority_countries}"
            plt_priority_countries.savefig(temp_priority_countries_path) # Save locally first
            uploaded_priority_countries = upload(temp_priority_countries_path)
            priority_countries_img_url = uploaded_priority_countries['secure_url']
            os.remove(temp_priority_countries_path)


            # Plot top publiction countries
            plt_countries = analyzer.plot_top_10_countries()
            filename_countries = f"top_countries_{filename_suffix}.png"
            temp_countries_path = f"/tmp/{filename_countries}"
            plt_countries.savefig(temp_countries_path) # Save locally first
            uploaded_countries = upload(temp_countries_path)
            countries_img_url = uploaded_countries['secure_url']
            os.remove(temp_countries_path)
 
            # Origin and Destination Countries
            plt_origin_destcountr = analyzer.analyze_patent_flow(top_n=10)
            filename_origin_destcountr = f"origin_destcountr_{filename_suffix}.png"
            temp_origin_destcountr_path = f"/tmp/{filename_origin_destcountr}"
            plt_origin_destcountr.savefig(temp_origin_destcountr_path) # Save locally first
            uploaded_origin_destcountr = upload(temp_origin_destcountr_path)
            origin_destcountr_img_url = uploaded_origin_destcountr['secure_url']
            os.remove(temp_origin_destcountr_path)

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
            filename_wcld_nouns = f"wcld_nouns_{filename_suffix}.png"
            temp_wcld_nouns_path = f"/tmp/{filename_wcld_nouns}"
            plt_wcld_nouns.savefig(temp_wcld_nouns_path) # Save locally first
            uploaded_wcld_nouns = upload(temp_wcld_nouns_path)
            wcld_nouns_img_url = uploaded_wcld_nouns['secure_url']
            os.remove(temp_wcld_nouns_path)

            # Plot wordcloud verbs
            plt_wcld_verbs = analyzer.generate_wordclouds_by_pos(pospeech = 'Verbs')
            filename_wcld_verbs = f"wcld_verbs_{filename_suffix}.png"
            temp_wcld_verbs_path = f"/tmp/{filename_wcld_verbs}"
            plt_wcld_verbs.savefig(temp_wcld_verbs_path) # Save locally first
            uploaded_wcld_verbs = upload(temp_wcld_verbs_path)
            wcld_verbs_img_url = uploaded_wcld_verbs['secure_url']
            os.remove(temp_wcld_verbs_path)


            # Plot wordcloud adjectives
            plt_wcld_adjectives = analyzer.generate_wordclouds_by_pos(pospeech = 'Adjectives')
            filename_wcld_adjectives = f"wcld_adjectives_{filename_suffix}.png"

            temp_wcld_adjectives_path = f"/tmp/{filename_wcld_adjectives}"
            plt_wcld_adjectives.savefig(temp_wcld_adjectives_path) # Save locally first
            uploaded_wcld_adjectives = upload(temp_wcld_adjectives_path)
            wcld_adjectives_img_url = uploaded_wcld_adjectives['secure_url']
            os.remove(temp_wcld_adjectives_path)
    
            extra_context = {
                'analysis1': "<p>Analysis complete. See the graphs below.</p>",
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


class IPC_ApplicantsView(FormView):
    template_name = "pages/IPC_Applicants.html"
    form_class = ReducedExcelUploadForm
    
    def form_valid(self, form):
        excel_file = form.cleaned_data.get('excel_file')
        ipc_list = None
        start_year = form.cleaned_data.get('start_year')
        end_year = form.cleaned_data.get('end_year')
        time_range = [start_year, end_year]
        print(f"Excel file received: {excel_file}")
        
        
        try:
            print("Initializing PatentNetwork...")
            analyzer = Patent_Analysis(excel_file)

            # Filter the data
            analyzer.filter_by_ipc_and_year(ipc_list, time_range)

            # Define image save path
            base_path = os.path.join(settings.MEDIA_ROOT, "images")
            os.makedirs(base_path, exist_ok=True)

   
            # Plot top IPCs
            plt_ipcs = analyzer.plot_top_ipcs()
            filename_ipcs = f"top_ipcs_{filename_suffix}.png"

            temp_ipcs_path = f"/tmp/{filename_ipcs}"
            plt_ipcs.savefig(temp_ipcs_path) # Save locally first
            uploaded_ipcs = upload(temp_ipcs_path)
            ipcs_img_url = uploaded_ipcs['secure_url']
            os.remove(temp_ipcs_path)

            # Show the boring table with IPC Groups and their definitions
            plt_defs = analyzer.get_top_ipcs_with_titles()
            filename_defs = f"top_ipcs_defs_{filename_suffix}.png"

            temp_defs_path = f"/tmp/{filename_defs}"
            plt_defs.savefig(temp_defs_path) # Save locally first
            defs_countries = upload(temp_defs_path)
            uploaded_defs = upload(temp_defs_path)
            defs_img_url = uploaded_defs['secure_url']
            os.remove(temp_defs_path)

            # Plot parallel coordinates
            plt_parallel = analyzer.plot_parallel_coordinates(top_n=5, year_range=(start_year, end_year + 1))
            filename_parallel = f"parallel_coordinates_{filename_suffix}.png"
            temp_parallel_path = f"/tmp/{filename_parallel}"
            plt_parallel.savefig(temp_parallel_path) # Save locally first
            uploaded_parallel = upload(temp_parallel_path)
            parallel_img_url = uploaded_parallel['secure_url']
            os.remove(temp_parallel_path)

            # Plot top Applicants
            plt_topAppl = analyzer.get_top_non_inventor_applicants(top_n=20)
            filename_topAppl = f"Top_Applicants{filename_suffix}.png"
            temp_topAppl_path = f"/tmp/{filename_topAppl}"
            plt_topAppl.savefig(temp_topAppl_path) # Save locally first
            uploaded_topAppl = upload(temp_topAppl_path)
            topAppl_img_url = uploaded_topAppl['secure_url']
            os.remove(temp_topAppl_path)

            # Applicants vs IPC bubble plot
            plt_Appl_IPC = analyzer.plot_applicant_ipc_bubble_chart(top_n=20)
            filename_Appl_IPC = f"Appl_IPC{filename_suffix}.png"
            temp_Appl_IPC_path = f"/tmp/{filename_Appl_IPC}"
            plt_Appl_IPC.savefig(temp_Appl_IPC_path) # Save locally first
            uploaded_Appl_IPC = upload(temp_Appl_IPC_path)
            Appl_IPC_img_url = uploaded_Appl_IPC['secure_url']
            os.remove(temp_Appl_IPC_path)

            # Plot top 5 Applicants parallel coordinates
            plt_ParalleltopAppl = analyzer.plot_applicant_parallel_coordinates(top_n=5, year_range=(start_year, end_year + 1))
            filename_ParalleltopAppl = f"ParalleltopAppl{filename_suffix}.png"
            temp_ParalleltopAppl_path = f"/tmp/{filename_ParalleltopAppl}"
            plt_ParalleltopAppl.savefig(temp_ParalleltopAppl_path) # Save locally first
            uploaded_ParalleltopAppl = upload(temp_ParalleltopAppl_path)
            ParalleltopAppl_img_url = uploaded_ParalleltopAppl['secure_url']
            os.remove(temp_ParalleltopAppl_path)
            

            extra_context = {
                'analysis1': "<p>Analysis complete. See the graphs below.</p>",
                'top_ipcs_img': ipcs_img_url,
                'top_ipcs_defs_img':defs_img_url,
                'parallel_img': parallel_img_url, 
                'topAppl_img': topAppl_img_url,
                'ParalleltopApp_img':ParalleltopAppl_img_url,
                'Appl_IPC_img': Appl_IPC_img_url
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
            filename_network = f"network_{filename_suffix}.png"
            temp_network_path = f"/tmp/{filename_network}"
            plt_network.savefig(temp_network_path) # Save locally first
            uploaded_network = upload(temp_network_path)
            network_img_url = uploaded_network['secure_url']
            os.remove(temp_network_path)
            
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
