"""
preprocessing.py
----------------
Cleans raw sensor data so the model receives trustworthy input.

Real IoT data is messy: sensors drop packets (missing values), glitch
(impossible readings) and send the same message twice (duplicates).
"""

import pandas as pd

from src import config

SENSOR_COLUMNS = [
    config.COL_AIR_TEMP, config.COL_PROCESS_TEMP,
    config.COL_RPM, config.COL_TORQUE, config.COL_TOOL_WEAR,
]

# Physically possible ranges. Anything outside = sensor glitch.
VALID_RANGES = {
    config.COL_AIR_TEMP: (250, 350),       # Kelvin
    config.COL_PROCESS_TEMP: (250, 370),
    config.COL_RPM: (0, 5000),
    config.COL_TORQUE: (0, 120),
    config.COL_TOOL_WEAR: (0, 400),
}


def clean_data(df):
    """Return a cleaned copy of the dataset and print what was fixed."""
    df = df.copy()
    start_rows = len(df)

    # 1) Remove exact duplicate messages
    df = df.drop_duplicates()

    # 2) Drop leakage / ID columns (failure-mode flags reveal the answer!)
    df = df.drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns])

    # 3) Replace impossible sensor readings with NaN so they get imputed below
    glitches = 0
    for col, (low, high) in VALID_RANGES.items():
        bad = ~df[col].between(low, high)
        glitches += int(bad.sum())
        df.loc[bad, col] = pd.NA

    # 4) Fill missing sensor values with the column median (robust to outliers)
    missing = int(df[SENSOR_COLUMNS].isnull().sum().sum())
    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col]).fillna(df[col].median())

    # 5) Make sure the target is 0/1 integer
    df[config.TARGET] = df[config.TARGET].astype(int)

    print(f"[CLEAN] Rows before: {start_rows:,} | after: {len(df):,} "
          f"| duplicates removed: {start_rows - len(df)}")
    print(f"[CLEAN] Sensor glitches fixed: {glitches} | missing values filled: {missing}")
    print(f"[CLEAN] Columns kept: {list(df.columns)}")
    return df
