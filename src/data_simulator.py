"""
data_simulator.py
-----------------
VIRTUAL IoT SENSOR SIMULATOR

We have no real factory, so we *simulate* one. This module generates sensor
readings from a fleet of milling machines, using the same physics-based rules
that were used to build the public UCI "AI4I 2020 Predictive Maintenance"
dataset (Matzka, 2020). Each row = one machine cycle reported by an IoT gateway.

Sensors simulated
    Air temperature [K]      - ambient temperature around the machine
    Process temperature [K]  - temperature inside the machining process
    Rotational speed [rpm]   - spindle speed
    Torque [Nm]              - load on the spindle
    Tool wear [min]          - minutes the cutting tool has been used

Failure modes (a machine fails if ANY of these happen)
    TWF  Tool Wear Failure       - worn tool breaks (wear 200-240 min)
    HDF  Heat Dissipation Failure- too little cooling (temp diff < 8.6 K AND rpm < 1380)
    PWF  Power Failure           - power outside 3500-9000 W
    OSF  Overstrain Failure      - tool wear x torque above a limit for the product type
    RNF  Random Failure          - 0.1% random failure (unpredictable, like real life)
"""

import numpy as np
import pandas as pd

from src import config


def _random_walk(rng, n, target_mean, target_std):
    """Smoothly drifting signal (like real temperature), rescaled to mean/std."""
    walk = np.cumsum(rng.normal(0, 1, n))
    walk = (walk - walk.mean()) / walk.std()
    return target_mean + walk * target_std


def simulate_sensor_data(n_samples=config.N_SAMPLES, seed=config.RANDOM_STATE):
    """Generate a realistic predictive-maintenance dataset and return a DataFrame."""
    rng = np.random.default_rng(seed)

    # 1) Product quality type: L (low) 50%, M (medium) 30%, H (high) 20%
    types = rng.choice(["L", "M", "H"], size=n_samples, p=[0.5, 0.3, 0.2])

    # 2) Temperatures: air drifts around 300 K, process runs ~10 K hotter
    air_temp = _random_walk(rng, n_samples, 300.0, 2.0)
    process_temp = air_temp + 10.0 + _random_walk(rng, n_samples, 0.0, 1.0)

    # 3) Torque and speed are linked: a heavier load slows the spindle down
    z = rng.normal(0, 1, n_samples)
    torque = np.clip(40 + 10 * z, 3.8, None)
    rpm = 1500 - 160 * z + rng.gamma(2.0, 20.0, n_samples)   # gamma -> right skew
    # Occasional load spikes (jammed workpiece / hard material): torque jumps, spindle slows
    spike = rng.random(n_samples) < 0.012
    torque[spike] += rng.uniform(15, 35, spike.sum())
    rpm[spike] -= rng.uniform(100, 250, spike.sum())
    rpm = np.clip(rpm, 1168, 2886)

    # 4) Tool wear grows every cycle; higher quality types wear the tool faster.
    #    The tool is replaced at a random point between 200 and 250 minutes.
    wear_step = {"H": 5, "M": 3, "L": 2}
    tool_wear = np.zeros(n_samples)
    current_wear, replace_at = 0.0, rng.uniform(200, 250)
    twf = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        current_wear += wear_step[types[i]]
        # Tool wear failure: a worn tool can break before it is replaced
        if 200 <= current_wear <= 240 and rng.random() < 0.05:
            twf[i] = 1
        tool_wear[i] = current_wear
        if twf[i] or current_wear >= replace_at:
            current_wear, replace_at = 0.0, rng.uniform(200, 250)

    # 5) Physics-based failure rules
    temp_diff = process_temp - air_temp
    power_w = torque * rpm * 2 * np.pi / 60          # P = torque x angular speed
    osf_limit = pd.Series(types).map({"L": 11000, "M": 12000, "H": 13000}).to_numpy()

    hdf = ((temp_diff < 8.6) & (rpm < 1380)).astype(int)
    pwf = ((power_w < 3500) | (power_w > 9000)).astype(int)
    osf = (tool_wear * torque > osf_limit).astype(int)
    rnf = (rng.random(n_samples) < 0.001).astype(int)
    machine_failure = ((twf + hdf + pwf + osf + rnf) > 0).astype(int)

    df = pd.DataFrame({
        "UDI": np.arange(1, n_samples + 1),
        "Product ID": [f"{t}{rng.integers(10000, 99999)}" for t in types],
        config.COL_TYPE: types,
        config.COL_AIR_TEMP: air_temp.round(1),
        config.COL_PROCESS_TEMP: process_temp.round(1),
        config.COL_RPM: rpm.round().astype(int),
        config.COL_TORQUE: torque.round(1),
        config.COL_TOOL_WEAR: tool_wear.astype(int),
        config.TARGET: machine_failure,
        "TWF": twf, "HDF": hdf, "PWF": pwf, "OSF": osf, "RNF": rnf,
    })
    return df


def save_simulated_data(df, path=config.SIMULATED_DATASET_FILE):
    config.ensure_directories()
    df.to_csv(path, index=False)
    print(f"[SIMULATOR] Saved {len(df):,} sensor records -> {path}")


if __name__ == "__main__":
    data = simulate_sensor_data()
    save_simulated_data(data)
    print(data.head())
    print("\nFailure rate: {:.2%}".format(data[config.TARGET].mean()))
    print(data[config.FAILURE_MODES].sum())
