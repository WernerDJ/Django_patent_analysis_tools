"""
Created on Tue Apr  1 11:57:44 2025

@author: werner
"""
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
from collections import Counter
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from datetime import datetime  
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from wordcloud import WordCloud
import networkx as nx
import io
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

#Import English stopwords from the nltk repository downloaded to the hard drive
nltk.data.path.append("/usr/share/nltk_data")


#Divide the word tokens into the three main syntactic groups
pos_groups = {
    'Nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
    'Verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'Adjectives': ['JJ', 'JJR', 'JJS']
}

class Patent_Analysis:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_excel(filepath, skiprows=5)
        self.data['MainIPC'] = self.data['I P C'].apply(self.extract_main_ipc)
        self.data['PubYear'] = pd.to_datetime(
            self.data['Publication Date'], format='%d.%m.%Y', errors='coerce'
        ).dt.year
        self.filtered_data = self.data.copy()
    
    def load_stopwords(self, input_txt_path='pages/stopwords.txt'):
        stopwords_set = set(nltk.corpus.stopwords.words('english'))
        try:
            with open(input_txt_path, "r", encoding="utf-8") as f:
                include_stopwords = {line.strip() for line in f}
            stopwords_set |= include_stopwords
        except FileNotFoundError:
            print(f"Warning: Stopwords file not found at {input_txt_path}")
        return stopwords_set

    def plot_top_10_countries(self):
        if self.filtered_data  is None:
            raise ValueError("Data not loaded. Please load the Excel file first.")
        
        # Exclude EP and WO
        country_counts = self.filtered_data [~self.filtered_data ['Country'].isin(['EP', 'WO'])]['Country'].value_counts()
        top_countries = country_counts.head(10)

        # Plot
        plt.figure(figsize=(10, 6))
        top_countries.plot(kind='bar', color='skyblue')
        plt.title("Top 10 Most Frequent Publication Countries (Excluding EP and WO)")
        plt.xlabel("Country")
        plt.ylabel("Number of Publications")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return plt
 
    def extract_priority_info(self, priority_data):
        """Extract the earliest priority year and corresponding country from priority data."""
        if pd.isna(priority_data):
            return None, None

        earliest_year = None
        earliest_country = None
        entries = priority_data.split(';')

        for entry in entries:
            entry = entry.strip()
            match = re.search(r'(?:\S+\s+)?(\d{2}\.\d{2}\.(\d{4}))\s+(\w+)', entry)
            if match:
                year = int(match.group(2))
                country = match.group(3)
            else:
                match = re.search(r'(?:\S+\s+)?(\d{4})-(\d{2})-(\d{2})\s+(\w+)', entry)
                if match:
                    year = int(match.group(1))
                    country = match.group(4)
                else:
                    match = re.search(r'.*?(\d{4})[-./].*?\s+(\w+)$|.*?(\w+)\s+(\d{4})[-./]', entry)
                    if match:
                        if match.group(1) and match.group(2):
                            year = int(match.group(1))
                            country = match.group(2)
                        elif match.group(3) and match.group(4):
                            country = match.group(3)
                            year = int(match.group(4))
                    else:
                        continue

            if year is not None and (earliest_year is None or year < earliest_year):
                earliest_year = year
                earliest_country = country

        return earliest_year, earliest_country

    def prepare_priority_data(self):
        """Prepare DataFrame with one row per unique patent family, containing earliest year and country."""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            raise ValueError("No data available for analysis")

        priority_groups = self.filtered_data.groupby('Priorities Data')
        priority_info = []

        for _, group in priority_groups:
            priority_data = group['Priorities Data'].iloc[0]
            year, country = self.extract_priority_info(priority_data)
            if year is not None and country is not None:
                priority_info.append((year, country))

        return pd.DataFrame(priority_info, columns=['Year', 'Country'])

    def plot_priority_countries_bar(self, priority_df):
        """Plot a bar chart of the Top 10 Priority Countries (deduplicated families)."""
        top_countries = priority_df['Country'].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_countries.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Top 10 Priority Countries")
        ax.set_xlabel("Country")
        ax.set_ylabel("Number of Priorities")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()

        return fig

    def plot_priority_years_bar(self, priority_df):
        """Plot a bar chart of frequency of priorities by year (deduplicated families)."""
        year_counts = priority_df.groupby('Year').size().sort_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        year_counts.plot(kind='bar', color='lightgreen', ax=ax)
        ax.set_title("Priority Years Frequency")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Priorities")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()

        return fig

    
    def analyze_patent_flow(self, top_n=10):
        """
        Analyzes patent flow between origin countries (from Priorities Data) and 
        destination countries (from Country column).
        
        - Extracts priority country codes from Priorities Data
        - Ignores rows where priority country equals publication country
        - Ignores rows where publication country or origin country is WO or EP
        - Creates a flow diagram showing patent movement from top N origin countries to destination countries
        
        Args:
            top_n: Number of top origin countries to include in the visualization (default: 10)
        """
        # Create a copy of filtered data to work with
        df = self.filtered_data.copy()
        
        # Extract the origin country (priority country) from Priorities Data column
        def extract_priority_country(priorities_data):
            if pd.isna(priorities_data) or priorities_data == '':
                return None
            
            # Extract all country codes from priorities data
            # Format is typically: "number date country; number date country"
            matches = re.findall(r'[0-9]+ [0-9\.]+\s+([A-Z]{2})', str(priorities_data))
            if matches:
                return matches[0]  # Return the first priority country code
            return None
        
        # Apply the function to extract origin countries
        df['Origin'] = df['Priorities Data'].apply(extract_priority_country)
        
        # Rename the Country column to Destination for clarity
        df['Destination'] = df['Country']
        
        # Filter the data according to requirements
        filtered_df = df[
            (df['Origin'].notna()) &  # Priority data is not empty
            (df['Origin'] != df['Destination']) &  # Priority country is different from publication country
            (~df['Origin'].isin(['WO', 'EP'])) &  # Origin country is not WO or EP
            (~df['Destination'].isin(['WO', 'EP']))  # Publication country is not WO or EP
        ]
        
        # Get the top N origin countries by total count
        origin_total_counts = filtered_df['Origin'].value_counts().head(top_n)
        top_origins = origin_total_counts.index.tolist()
        
        # Filter data to include only top origin countries
        filtered_df = filtered_df[filtered_df['Origin'].isin(top_origins)]
        
        # Count the frequency of each origin-destination pair
        flow_df = filtered_df.groupby(['Origin', 'Destination']).size().reset_index(name='Count')
        
        # Count the total occurrences of each origin and destination country
        origin_counts = flow_df.groupby('Origin')['Count'].sum().reset_index()
        dest_counts = flow_df.groupby('Destination')['Count'].sum().reset_index()
        
        # Create the visualization
        figure = self._create_patent_flow_diagram(flow_df, origin_counts, dest_counts, top_n)
        
        return figure
    
    def _create_patent_flow_diagram(self, flow_df, origin_counts, dest_counts, top_n=10):
        """
        Creates a flow diagram showing patent movement from origin to destination countries.
        
        Args:
            flow_df: DataFrame with Origin, Destination, and Count columns
            origin_counts: DataFrame with count of patents per origin country
            dest_counts: DataFrame with count of patents per destination country
            top_n: Number of top origin countries included in the analysis
        """
        # Create a figure with a single axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Import required libraries if not already imported
        from matplotlib.path import Path
        import matplotlib.patches as patches
        
        # Define colors for countries
        unique_countries = pd.concat([
            pd.Series(flow_df['Origin'].unique()), 
            pd.Series(flow_df['Destination'].unique())
        ]).unique()
        
        # Create a color map for countries
        color_map = {}
        cmap = plt.cm.tab20
        for i, country in enumerate(unique_countries):
            color_map[country] = cmap(i % 20)
        
        # Set up the positions for the bars
        left_x = 0
        right_x = 1
        
        # Sort countries by frequency
        origin_counts = origin_counts.sort_values('Count', ascending=False)
        dest_counts = dest_counts.sort_values('Count', ascending=False)
        
        # Calculate the total for normalization
        total_origin = origin_counts['Count'].sum()
        total_dest = dest_counts['Count'].sum()
        
        # Prepare the positions and heights for the origin bars
        origin_positions = {}
        y_offset = 0
        for _, row in origin_counts.iterrows():
            country = row['Origin']
            count = row['Count']
            height = count / total_origin
            origin_positions[country] = (y_offset, height)
            
            # Draw the origin bar
            ax.add_patch(plt.Rectangle(
                (left_x, y_offset), 
                0.1, 
                height, 
                color=color_map[country], 
                alpha=0.8
            ))
            
            # Add label for origin country - moved outside the bar
            if height > 0.01:
                ax.text(
                    left_x - 0.01, 
                    y_offset + height/2, 
                    country, 
                    ha='right', 
                    va='center', 
                    fontsize=9
                )
                
            y_offset += height
        
        # Prepare the positions and heights for the destination bars
        dest_positions = {}
        y_offset = 0
        for _, row in dest_counts.iterrows():
            country = row['Destination']
            count = row['Count']
            height = count / total_dest
            dest_positions[country] = (y_offset, height)
            
            # Draw the destination bar
            ax.add_patch(plt.Rectangle(
                (right_x - 0.1, y_offset), 
                0.1, 
                height, 
                color=color_map[country], 
                alpha=0.8
            ))
            
            # Add label for destination country - moved outside the bar
            if height > 0.01:
                ax.text(
                    right_x + 0.01, 
                    y_offset + height/2, 
                    country, 
                    ha='left', 
                    va='center', 
                    fontsize=9
                )
            
            y_offset += height
        
        # Draw the connecting lines between origin and destination
        for _, row in flow_df.iterrows():
            origin = row['Origin']
            dest = row['Destination']
            count = row['Count']
            
            # Get the positions
            origin_y = origin_positions[origin][0] + origin_positions[origin][1] / 2
            dest_y = dest_positions[dest][0] + dest_positions[dest][1] / 2
            
            # Calculate the line width based on the count
            line_width = 1 + 5 * (count / flow_df['Count'].max())
            
            # Calculate the transparency based on the count
            alpha = 0.2 + 0.6 * (count / flow_df['Count'].max())
            
            # Define the curve using control points
            verts = [
                (left_x + 0.1, origin_y),                   # Start point
                ((left_x + right_x) / 2, origin_y),         # Control point 1
                ((left_x + right_x) / 2, dest_y),           # Control point 2
                (right_x - 0.1, dest_y)                     # End point
            ]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            
            # Draw the path with the appropriate color and width
            patch = patches.PathPatch(
                path, 
                facecolor='none',
                edgecolor=color_map[origin],
                linewidth=line_width,
                alpha=alpha,
                zorder=1
            )
            ax.add_patch(patch)
        
        # Adjust the x limits to accommodate the labels
        ax.set_xlim(left_x - 0.1, right_x + 0.1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add labels for the bars
        ax.text(left_x + 0.05, 1.05, 'Origin Countries', ha='center', va='bottom', fontsize=12)
        ax.text(right_x - 0.05, 1.05, 'Destination Countries', ha='center', va='bottom', fontsize=12)
        
        # Add title above the graph
        plt.figtext(0.5, 0.95, f"Destination Countries of the top {top_n} priority countries", 
                     ha='center', fontsize=14, weight='bold')
        
        # Remove axis frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for the title
        return plt


    def extract_main_ipc(self, ipc_codes):
        if pd.isna(ipc_codes):
            return None
        codes = re.findall(r'\b[A-Z]\d{2}[A-Z]?', str(ipc_codes))
        if not codes:
            return None
        short_codes = [code[:4] for code in codes]
        counter = Counter(short_codes)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else short_codes[0]

    def filter_by_ipc_and_year(self, ipc_list=None, year_range=None):
        data = self.filtered_data
        if ipc_list:
            data = self.data[data['MainIPC'].isin(ipc_list)]
        if year_range and len(year_range) == 2:
            start, end = year_range
            data = data[(data['PubYear'] >= start) & (data['PubYear'] <= end)]
        self.filtered_data = data
        return data

    def plot_top_ipcs(self, top_n=10):
        all_ipcs = []
        for ipcs in self.filtered_data['I P C'].dropna():
            codes = re.findall(r'\b[A-Z]\d{2}[A-Z]?', str(ipcs))
            all_ipcs.extend(codes)
        ipc_counts = pd.Series(all_ipcs).value_counts().head(top_n)
        self.top_ipc_counts = ipc_counts  # Store for reuse in the ellaboration of a boring table
        plt.figure(figsize=(10, 6))
        ipc_counts.plot(kind='bar', color='lightgreen')
        plt.title(f"Top {top_n} Most Frequent IPCs")
        plt.xlabel("IPC Code")
        plt.ylabel("Number of Publications")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return plt
        
   
    def get_top_ipcs_with_titles(self, title_file="cpc_group_titles.csv"):
        """
        Uses stored top IPC counts and enriches them with titles from the CPC definition CSV.
        Then creates and saves a PNG image containing the table.
        """
        if not hasattr(self, 'top_ipc_counts'):
            raise ValueError("Top IPCs not calculated yet. Please call plot_top_ipcs() first.")
    
        # Load IPC titles
        current_dir = os.path.dirname(os.path.abspath(__file__))
        title_path = os.path.join(current_dir, title_file)
    
        if not os.path.isfile(title_path):
            raise FileNotFoundError(f"Title file '{title_file}' not found at {title_path}.")
    
        df_titles = pd.read_csv(title_path)
        title_dict = dict(zip(df_titles['Group'], df_titles['Title']))
    
        # Create a new figure without using matplotlib's table
        fig, ax = plt.subplots(figsize=(12, 0.5 * (len(self.top_ipc_counts) + 1)))
        ax.axis('off')
        
        # Remove all margins and make axes fill figure
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Create our own custom table layout
        header = ["IPC Group", "Count", "Title"]
        rows = []
        for ipc in self.top_ipc_counts.index:
            count = int(self.top_ipc_counts[ipc])
            title = title_dict.get(ipc, "No title found")
            rows.append([ipc, count, title])
        
        # Calculate positions
        row_height = 0.8 / (len(rows) + 1)
        font_size = 10
        
        # Draw header
        for i, col_name in enumerate(header):
            x_pos = 0.02 if i == 0 else (0.17 if i == 1 else 0.27)  # Shifted left
            y_pos = 1 - row_height * 0.5
            ax.text(x_pos, y_pos, col_name, fontsize=font_size, weight='bold', 
                    fontfamily='monospace', transform=ax.transAxes)
        
        # Draw row lines
        ax.axhline(y=1 - row_height, xmin=0.02, xmax=0.98, color='black', linewidth=1)
        
        # Draw data rows
        for row_idx, row_data in enumerate(rows):
            y_pos = 1 - (row_idx + 1.5) * row_height
            
            # IPC Group column
            ax.text(0.02, y_pos, row_data[0], fontsize=font_size, 
                    fontfamily='monospace', transform=ax.transAxes)
            
            # Count column
            ax.text(0.17, y_pos, str(row_data[1]), fontsize=font_size, 
                    fontfamily='monospace', transform=ax.transAxes)
            
            # Title column - positioned with no extra space
            ax.text(0.27, y_pos, row_data[2], fontsize=font_size, 
                    fontfamily='monospace', transform=ax.transAxes)
            
            # Row separator line
            ax.axhline(y=1 - (row_idx + 2) * row_height, xmin=0.02, xmax=0.98, 
                       color='black', linewidth=0.5, alpha=0.3)
        
        # Draw vertical separator lines (moved left, removed right border)
        ax.axvline(x=0.02, ymin=0, ymax=1, color='black', linewidth=1)
        ax.axvline(x=0.17, ymin=0, ymax=1, color='black', linewidth=1)
        ax.axvline(x=0.27, ymin=0, ymax=1, color='black', linewidth=1)
        return plt

    def generate_wordclouds_by_pos(self, text_column='Abstract', pospeech='Nouns'):
        stopwords = self.load_stopwords()  # Load tstopwords
        text_data = self.filtered_data[text_column].dropna().astype(str).str.cat(sep=' ')
        tokens = word_tokenize(text_data.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stopwords]
        
        tagged_tokens = pos_tag(tokens)
        group_words = [word for word, tag in tagged_tokens if tag in pos_groups[pospeech]]
        word_freq = Counter(group_words)
        
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                                colormap='Dark2').generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'{pospeech} Word Cloud', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        return plt

    def plot_parallel_coordinates(self, top_n=5, year_range=None):
        if year_range is None:
            current_year = datetime.now().year
            year_range = range(current_year - 20, current_year)     

        df_filtered = self.filtered_data.copy()
        df_filtered = df_filtered[df_filtered['MainIPC'].notna()]
        df_filtered['PubYear'] = pd.to_numeric(df_filtered['PubYear'], errors='coerce').fillna(0).astype(int)
        df_filtered = df_filtered[df_filtered['PubYear'] > 0]

        ipc_counts = df_filtered['MainIPC'].value_counts()
        top_ipcs = ipc_counts.head(top_n).index

        ipc_yearly_counts = pd.DataFrame(index=top_ipcs, columns=year_range, data=0)
        for ipc in top_ipcs:
            ipc_data = df_filtered[df_filtered['MainIPC'] == ipc]
            yearly_counts = ipc_data['PubYear'].value_counts()
            for year in year_range:
                ipc_yearly_counts.at[ipc, year] = yearly_counts.get(year, 0)

        normalized_data = ipc_yearly_counts.div(ipc_yearly_counts.max(axis=1), axis=0)

        fig, host = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab10.colors

        for i, ipc in enumerate(top_ipcs):
            data = normalized_data.loc[ipc].values
            x = np.linspace(0, len(year_range) - 1, len(year_range))

            verts = [(x[0], data[0])]
            codes = [Path.MOVETO]
            for j in range(1, len(x)):
                x0, y0 = x[j - 1], data[j - 1]
                x1, y1 = x[j], data[j]
                ctrl1 = (x0 + (x1 - x0) / 3, y0)
                ctrl2 = (x0 + 2 * (x1 - x0) / 3, y1)
                verts.extend([ctrl1, ctrl2, (x1, y1)])
                codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=4, edgecolor=colors[i % len(colors)])
            host.add_patch(patch)

        host.set_xlim(0, len(year_range) - 1)
        host.set_xticks(range(len(year_range)))
        host.set_xticklabels(year_range, fontsize=10, rotation=45)
        host.set_title(
            f'Parallel Coordinates Plot: Top {top_n} IPC Codes ({year_range[0]}–{year_range[-1]})',
            fontsize=16
        )
        host.set_xlabel('Publication Year')
        host.set_ylabel('Normalized Patent Counts')
        plt.legend(top_ipcs, title='Main IPC', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return plt




class Patent_Network:
    def __init__(self, input_file):
        self.input_file = input_file
        self.data = pd.read_excel(input_file, skiprows=5)
        self.filtered_data = None
        self.graph = None

    def _clean_applicant_name(self, name):
        name = name.strip()
        if name.lower().endswith(', inc') or name.lower().endswith(', inc.'):
            name = name[:name.lower().rfind(', inc')].strip()
        return name

    def _has_similar_value(self, applicant, inventors):
        applicant = applicant.lower().strip()
        for inventor in inventors:
            inventor = inventor.lower().strip()
            if any(applicant[i:i+5] in inventor for i in range(len(applicant) - 4)):
                return True
        return False

    def filter_data(self):
        self.data['Applicants'] = self.data['Applicants'].fillna('').str.split(';')
        self.data['Inventors'] = self.data['Inventors'].fillna('').str.split(';')

        filtered_rows = []
        for _, row in self.data.iterrows():
            applicants = row['Applicants']
            inventors = row['Inventors']

            if any(applicant.strip().lower() == inventor.strip().lower()
                   for applicant in applicants for inventor in inventors):
                continue

            filtered_applicants = [
                self._clean_applicant_name(applicant)
                for applicant in applicants
                if not self._has_similar_value(applicant, inventors)
            ]

            selected_applicant = filtered_applicants[0] if filtered_applicants else None
            if selected_applicant:
                filtered_rows.append({
                    "Applicant": selected_applicant,
                    "Inventors": '; '.join(inventors)
                })

        filtered_df = pd.DataFrame(filtered_rows)
        
        # Compact by grouping applicants
        compacted_data = (
            filtered_df.groupby(filtered_df['Applicant'].str.lower())
            .agg({"Inventors": lambda x: '; '.join(set('; '.join(x).split('; ')))})
            .reset_index()
        )
        compacted_data.columns = ['Applicant', 'Inventors']
        self.filtered_data = compacted_data

    def build_graph(self):
        if self.filtered_data is None:
            raise ValueError("Data must be filtered first using filter_data()")

        G = nx.Graph()
        for i, row1 in self.filtered_data.iterrows():
            for j, row2 in self.filtered_data.iterrows():
                if i >= j:
                    continue
                inventors1 = set(row1['Inventors'].split('; '))
                inventors2 = set(row2['Inventors'].split('; '))
                shared_inventors = inventors1 & inventors2
                weight = len(shared_inventors)
                if weight > 0:
                    G.add_edge(row1['Applicant'], row2['Applicant'], weight=weight)
        self.graph = G
    
    def generate_network_image(self, top_n=10):
        """Generate a network visualization and return as base64 encoded string"""
        if self.graph is None:
            raise ValueError("Graph must be built first using build_graph()")

        degrees = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:top_n]
        top_nodes = [node for node, _ in degrees]
        H = self.graph.subgraph(top_nodes)

        # Create a Figure and FigureCanvas
        fig = Figure(figsize=(12, 12))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        
        pos = nx.circular_layout(H)
        nx.draw(
            H, pos, ax=ax, with_labels=True, node_size=2500, font_size=6,
            font_weight='bold', edge_color='gray', node_color='skyblue'
        )
        labels = nx.get_edge_attributes(H, 'weight')
        nx.draw_networkx_edge_labels(H, pos, edge_labels=labels, font_size=8, ax=ax)
        ax.set_title(f"Top {top_n} Most Connected Applicants", fontsize=16)
        
        # Return the figure object directly
        return fig
        
