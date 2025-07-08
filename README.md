# Machine Learning for Mpro Inhibitor Discovery

This project uses machine learning to identify potential inhibitors for the main protease (Mpro) of SARS-CoV-2, based on bioactivity data from the COVID Moonshot project. The workflow involves data downloading, preprocessing, featurization, and similarity analysis.

## Project Structure

- **/data**: Stores raw and processed datasets.
- **/notebooks**: Contains the original Jupyter Notebook for exploration.
- **/output**: Default location for saved plots and models.
- **/scripts**: Main executable script to run the entire workflow.
- **/src**: Modularized Python source code for each step of the pipeline.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aliammar0161/Mpro-inhibitor-discovery.git
    cd mpro-inhibitor-discovery
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Workflow

To execute the full data processing and analysis pipeline, run the main script from the root directory:

```bash
python scripts/run_workflow.py
