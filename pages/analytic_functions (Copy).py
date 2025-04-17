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



class Patent_Analysis:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_excel(filepath, skiprows=5)
        self.data['MainIPC'] = self.data['I P C'].apply(self.extract_main_ipc)
        self.data['PubYear'] = pd.to_datetime(
            self.data['Publication Date'], format='%d.%m.%Y', errors='coerce'
        ).dt.year
        self.filtered_data = self.data.copy()

    def plot_top_10_countries(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load the Excel file first.")
        
        # Exclude EP and WO
        country_counts = self.data[~self.data['Country'].isin(['EP', 'WO'])]['Country'].value_counts()
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
        data = self.data
        if ipc_list:
            data = data[data['MainIPC'].isin(ipc_list)]
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
        plt.figure(figsize=(10, 6))
        ipc_counts.plot(kind='bar', color='lightgreen')
        plt.title(f"Top {top_n} Most Frequent IPCs")
        plt.xlabel("IPC Code")
        plt.ylabel("Number of Publications")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
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
