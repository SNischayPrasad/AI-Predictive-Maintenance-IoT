"""
feature_engineering.py
----------------------
Turns raw sensor readings into features that describe machine *health*.

A maintenance engineer does not look only at "torque = 60 Nm". They ask:
  - How much power is the spindle drawing?      -> Power [W]
  - Is the machine cooling itself properly?     -> Temp difference [K]
  - How stressed is the cutting tool?           -> Strain [minNm]
We encode that domain knowledge as new columns.
"""

import numpy as np

from src import config

TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}   # ordinal: quality goes L < M < H


def add_features(df):
    """Return a copy of df with engineered features added."""
    df = df.copy()

    # Encode product type as a number (models need numbers, not letters)
    df["Type_encoded"] = df[config.COL_TYPE].map(TYPE_ENCODING)

    # Mechanical power = torque x angular velocity (rad/s)
    df["Power [W]"] = df[config.COL_TORQUE] * df[config.COL_RPM] * 2 * np.pi / 60

    # Heat dissipation: small difference = poor cooling
    df["Temp difference [K]"] = df[config.COL_PROCESS_TEMP] - df[config.COL_AIR_TEMP]

    # Overstrain indicator: worn tool under heavy load
    df["Strain [minNm]"] = df[config.COL_TOOL_WEAR] * df[config.COL_TORQUE]

    df = df.drop(columns=[config.COL_TYPE])
    return df


def get_feature_columns(df):
    """All columns except the target are model inputs."""
    return [c for c in df.columns if c != config.TARGET]
