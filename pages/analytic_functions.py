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
from datetime import datetime  #Added import
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from wordcloud import WordCloud
nltk.data.path.append("/usr/share/nltk_data")

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
        plt.title("Top 10 Most Frequent Countries (Excluding EP and WO)")
        plt.xlabel("Country")
        plt.ylabel("Number of Publications")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
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
