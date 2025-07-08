import sys
import os
import warnings

# This allows the script to find and import modules from the 'src' directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules from the 'src' directory
from src.data_utils import download_data, load_and_preprocess_data
from src.featurization import get_morgan_fingerprints
from src.visualization import plot_umap
from src.modeling import print_and_plot_metrics 

#Import necessary third-party libraries
import deepchem as dc
import xgboost as xgb 

# --- Suppress known warnings for cleaner output ---
warnings.filterwarnings(
    "ignore",
    message="No normalization for SPS. Feature removed!",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="No normalization for AvgIpc. Feature removed!",
    category=UserWarning,
)


def main():
    """
    Main function to execute the entire Mpro inhibitor discovery workflow.
    """
    print("--- Starting Mpro Inhibitor Discovery Workflow ---")

    # --- Configuration ---
    DATA_URL = "https://covid.postera.ai/covid/activity_data.csv"
    OUTPUT_DIR = "output"
    DATA_DIR = "data/raw"

    # --- Step 1: Data Loading and Preprocessing ---
    print("\n[1/3] Loading and Preprocessing Data...")
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_data_path = os.path.join(DATA_DIR, "activity_data.csv")
    download_data(DATA_URL, save_path=raw_data_path)
    df = load_and_preprocess_data(filepath=raw_data_path)
    print(f"Loaded {len(df)} compounds after preprocessing.")

    # --- Step 2: Featurization ---
    print("\n[2/3] Generating Molecular Fingerprints...")
    features = get_morgan_fingerprints(df["SMILES"].tolist())
    # The DiskDataset is efficient for handling large datasets that may not fit in memory.
    dataset = dc.data.DiskDataset.from_numpy(
        X=features, y=df["labels"].values, ids=df["SMILES"].values
    )
    print("Featurization complete.")


    # UMAP Visualization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    umap_save_path = os.path.join(OUTPUT_DIR, "umap_projection.png")
    plot_umap(
        features,
        df["labels"].values,
        "UMAP Projection of the Moonshot Dataset",
        save_path=umap_save_path
    )
    print("Analysis complete.")

    # --- Step 3: Modeling  ---
    print("\n[3/3] Model Training and Evaluation...")
    print("  - Splitting data using Scaffold Splitter...")
    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, _, test_dataset = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0, frac_test=0.2
    )
    
    print("  - Training XGBoost model...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(train_dataset.X, train_dataset.y)
    
    print("  - Evaluating model performance:")
    y_pred = model.predict(test_dataset.X)
    y_prob_pred = model.predict_proba(test_dataset.X)[:, 1]
    print_and_plot_metrics(test_dataset.y, y_pred, y_prob_pred)

    print("\n--- Workflow Finished Successfully ---")


if __name__ == "__main__":
    main()