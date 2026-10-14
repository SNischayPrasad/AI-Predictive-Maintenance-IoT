"""
iot_stream_simulator.py
-----------------------
REAL-TIME VIRTUAL MACHINE SIMULATION

Pretends to be one milling machine on a factory floor that sends a sensor
packet every cycle. The trained model scores each packet live, the alert
system reacts, and when a CRITICAL alert repeats the "maintenance team"
services the machine (tool replaced, fault fixed).

Scripted story (so the demo is repeatable):
  * Tool wears continuously -> torque slowly rises as the tool gets dull
  * Cycles  70-110 : cooling fan degrades  (heat dissipation fault)
  * Cycles 250-265 : motor overload        (power fault)
  * Tool wear keeps building up             (tool wear / overstrain risk)
"""

import time

import numpy as np
import pandas as pd

from src import config
from src.alert_system import format_alert, generate_alert
from src.model_training import load_model
from src.predict import build_reading, predict_failure

PRODUCT_TYPE = "L"
WEAR_PER_CYCLE = 2
CONSECUTIVE_CRITICAL_FOR_MAINTENANCE = 3


def _sensor_packet(rng, cycle, tool_wear, cooling_fault, overload):
    """Generate one realistic sensor reading for the current machine state."""
    air = 300 + 1.5 * np.sin(cycle / 40) + rng.normal(0, 0.15)      # day/night drift
    temp_gap = 10.2 + rng.normal(0, 0.3)
    rpm = 1500 + rng.normal(0, 25)
    torque = 38 + 0.06 * tool_wear + rng.normal(0, 1.5)              # dull tool -> more force

    if cooling_fault:                       # fan failing -> machine runs hotter, slows down
        temp_gap -= cooling_fault
        rpm -= 160
        torque += 6
    if overload:                            # jammed workpiece -> huge torque
        torque += 32
        rpm -= 220

    return dict(product_type=PRODUCT_TYPE, air_temp=round(air, 1),
                process_temp=round(air + temp_gap, 1), rpm=int(rpm),
                torque=round(torque, 1), tool_wear=int(tool_wear))


def run_stream_simulation(n_cycles=320, delay=0.0, verbose=True, seed=7):
    bundle = load_model()
    rng = np.random.default_rng(seed)
    tool_wear, cooling_fixed, overload_fixed = 0, False, False
    critical_streak, rows = 0, []

    print(f"\n[STREAM] Virtual machine VM-07 online. Model: {bundle['model_name']}")
    print(f"[STREAM] Streaming {n_cycles} sensor packets...\n")

    for cycle in range(1, n_cycles + 1):
        # Fault severity grows over time until maintenance fixes it
        cooling_fault = 0.0
        if 70 <= cycle <= 110 and not cooling_fixed:
            cooling_fault = min(2.5, (cycle - 70) * 0.12)
        overload = 250 <= cycle <= 265 and not overload_fixed

        packet = _sensor_packet(rng, cycle, tool_wear, cooling_fault, overload)
        scored = predict_failure(build_reading(**packet), bundle).iloc[0]
        prob = scored["failure_probability"]
        alert = generate_alert("VM-07", scored.to_dict(), prob, log=True,
                               log_file=config.ALERTS_LOG_FILE)

        event = ""
        critical_streak = critical_streak + 1 if alert["alert_level"] == "CRITICAL" else 0
        if critical_streak >= CONSECUTIVE_CRITICAL_FOR_MAINTENANCE:
            event = "MAINTENANCE"
            tool_wear = 0                              # new cutting tool
            if 70 <= cycle <= 110:
                cooling_fixed = True                   # fan replaced
            if 250 <= cycle <= 265:
                overload_fixed = True                  # jam cleared
            critical_streak = 0

        rows.append({"cycle": cycle,
                     "Air temperature [K]": packet["air_temp"],
                     "Process temperature [K]": packet["process_temp"],
                     "Rotational speed [rpm]": packet["rpm"],
                     "Torque [Nm]": packet["torque"],
                     "Tool wear [min]": packet["tool_wear"],
                     "failure_probability": prob,
                     "alert_level": alert["alert_level"],
                     "likely_causes": alert["likely_causes"],
                     "event": event})

        if verbose and (alert["alert_level"] != "NORMAL" or cycle % 25 == 0 or event):
            print(f"cycle {cycle:>3} | wear {packet['tool_wear']:>3} min | "
                  f"torque {packet['torque']:>5.1f} Nm | rpm {packet['rpm']:>4} | "
                  + format_alert(alert))
            if event:
                print(f"          >>> MAINTENANCE PERFORMED at cycle {cycle} "
                      f"(tool replaced, faults cleared) <<<")
            time.sleep(delay)

        if event != "MAINTENANCE":
            tool_wear += WEAR_PER_CYCLE

    log_df = pd.DataFrame(rows)
    config.ensure_directories()
    log_df.to_csv(config.STREAM_LOG_FILE, index=False)

    summary = log_df["alert_level"].value_counts()
    print("\n[STREAM] Simulation summary:")
    print(summary.to_string())
    print(f"[STREAM] Maintenance actions triggered: {(log_df['event'] == 'MAINTENANCE').sum()}")
    print(f"[STREAM] Log saved -> {config.STREAM_LOG_FILE}")
    return log_df
