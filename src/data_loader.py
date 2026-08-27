import pandas as pd
from pathlib import Path

# Load data from CSV files

def load_data(data_path):
    summaries = pd.read_csv(
        Path(data_path) / "summaries_train.csv"
    )

    prompts = pd.read_csv(
        Path(data_path) / "prompts_train.csv"
    )

    return summaries, prompts