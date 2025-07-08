import os
import requests
import pandas as pd
import numpy as np

def download_data(url, save_path="data/raw/activity_data.csv"):
    """Downloads data if it doesn't already exist."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"Downloading data from {url} to {save_path}...")
        r = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print(f"Data file already exists at {save_path}.")

def load_and_preprocess_data(filepath="data/raw/activity_data.csv"):
    """Loads and preprocesses the dataset from the given filepath."""
    data = pd.read_csv(filepath)
    data_f = data[["SMILES", "f_avg_IC50"]].dropna()
    data_f_IC50 = data_f["f_avg_IC50"].astype(float)
    # Calculate pIC50, converting IC50 from µM to Molar (1e-6)
    data_f["pIC50"] = -1 * np.log10(data_f_IC50 / 1e6)
    # Create binary labels based on a pIC50 threshold of 6
    data_f["labels"] = np.where(data_f["pIC50"] <= 6, 0, 1)
    return data_f