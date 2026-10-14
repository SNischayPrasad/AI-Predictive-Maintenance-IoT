"""
data_loader.py
--------------
Loads the sensor dataset.

Priority:
  1. Real UCI AI4I 2020 dataset at data/raw/ai4i2020.csv (if you downloaded it)
  2. Previously simulated dataset at data/raw/simulated_sensor_data.csv
  3. Otherwise -> run the virtual IoT simulator to create one
"""

import pandas as pd

from src import config
from src.data_simulator import simulate_sensor_data, save_simulated_data


def load_dataset(force_simulate=False):
    """Return (DataFrame, source_name)."""
    config.ensure_directories()

    if config.REAL_DATASET_FILE.exists() and not force_simulate:
        df = pd.read_csv(config.REAL_DATASET_FILE)
        source = "UCI AI4I 2020 (real benchmark dataset)"
    elif config.SIMULATED_DATASET_FILE.exists() and not force_simulate:
        df = pd.read_csv(config.SIMULATED_DATASET_FILE)
        source = "Virtual IoT simulator (cached)"
    else:
        df = simulate_sensor_data()
        save_simulated_data(df)
        source = "Virtual IoT simulator (freshly generated)"

    print(f"[LOAD] Source : {source}")
    print(f"[LOAD] Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df, source


def describe_dataset(df):
    """Print a quick overview - useful as a screenshot for your README."""
    print("\n[LOAD] First 5 rows:")
    print(df.head().to_string())
    print("\n[LOAD] Missing values per column:")
    print(df.isnull().sum().to_string())
    print(f"\n[LOAD] Failure rate: {df[config.TARGET].mean():.2%} "
          f"({df[config.TARGET].sum()} failures out of {len(df):,} cycles)")
    print("[LOAD] Failures by mode:")
    print(df[config.FAILURE_MODES].sum().to_string())
