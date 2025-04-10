"""
Created on Tue Apr  1 11:57:44 2025

@author: werner
"""
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
from collections import Counter

# Load the Excel file is done in the view so we don't do it here.

def extract_main_ipc(ipc_codes):
    if pd.isna(ipc_codes):
        return None
    # Extract all IPC codes
    codes = re.findall(r'\b[A-Z]\d{2}[A-Z]?', str(ipc_codes))
    if not codes:
        return None
    # Extract first 4 characters of each code
    short_codes = [code[:4] for code in codes]
    # Count occurrences
    counter = Counter(short_codes)
    most_common = counter.most_common(1)
    # Return most common if available, else return the first one
    return most_common[0][0] if most_common else short_codes[0]

def filter_data_by_ipc_and_year(data, ipc_list, year_range):
    # If IPCs list is empty, apply no filter; otherwise, filter by IPCs
    if ipc_list:
        data = data[data['MainIPC'].isin(ipc_list)]
    # If TimeRange is empty or invalid, apply no filter; otherwise, filter by PubYear
    if year_range and len(year_range) == 2:
        start_year, end_year = year_range
        data = data[(data['PubYear'] >= start_year) & (data['PubYear'] <= end_year)]
    return data

def plot_top_10_countries(data):
    # Exclude EP and WO
    country_counts = data[~data['Country'].isin(['EP', 'WO'])]['Country'].value_counts()
    top_countries = country_counts.head(10)
    # Plot
    plt.figure(figsize=(10, 6))
    top_countries.plot(kind='bar', color='skyblue')
    plt.title("Top 10 Most Frequent Countries (Excluding EP and WO)")
    plt.xlabel("Country")
    plt.ylabel("Number of Publications")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return plt

def plot_top_10_ipcs(data):
    # Extract IPCs from the 'I P C' column
    all_ipcs = []
    for ipcs in data['I P C'].dropna():
        codes = re.findall(r'\b[A-Z]\d{2}[A-Z]?', str(ipcs))
        all_ipcs.extend(codes)
    # Count frequencies
    ipc_counts = pd.Series(all_ipcs).value_counts()
    top_ipcs = ipc_counts.head(10)
    # Plot
    plt.figure(figsize=(10, 6))
    top_ipcs.plot(kind='bar', color='lightgreen')
    plt.title("Top 10 Most Frequent IPCs")
    plt.xlabel("IPC Code")
    plt.ylabel("Number of Publications")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return plt
