"""
alert_system.py
---------------
Turns a failure probability into an action a maintenance team understands.

    probability <  30%  -> NORMAL    (keep running)
    30% - 60%           -> WARNING   (schedule inspection)
    probability >= 60%  -> CRITICAL  (stop machine / immediate maintenance)

It also gives a likely root cause using simple engineering rules, so the
technician knows WHERE to look (tool, cooling, power, overload).
"""

import csv
from datetime import datetime

from src import config


def get_alert_level(probability):
    if probability >= config.CRITICAL_THRESHOLD:
        return "CRITICAL"
    if probability >= config.WARNING_THRESHOLD:
        return "WARNING"
    return "NORMAL"


def diagnose(reading):
    """Rule-based hints about WHY the risk is high. `reading` is a dict of features."""
    causes = []
    if reading["Tool wear [min]"] >= 200:
        causes.append("Tool nearly worn out -> replace cutting tool")
    if reading["Temp difference [K]"] < 8.6 and reading["Rotational speed [rpm]"] < 1380:
        causes.append("Poor heat dissipation -> check cooling system")
    if reading["Power [W]"] < 3500 or reading["Power [W]"] > 9000:
        causes.append("Power outside safe range (3.5-9 kW) -> check drive/motor")
    limit = {0: 11000, 1: 12000, 2: 13000}.get(int(reading["Type_encoded"]), 11000)
    if reading["Strain [minNm]"] > limit * 0.9:
        causes.append("Tool overstrain -> reduce load or replace tool")
    return causes or ["No single dominant cause - inspect machine"]


ACTIONS = {
    "NORMAL": "Continue normal operation",
    "WARNING": "Schedule inspection within next shift",
    "CRITICAL": "STOP machine and perform maintenance immediately",
}


def generate_alert(machine_id, reading, probability, log=True, log_file=config.ALERTS_LOG_FILE):
    """Build an alert dict; append WARNING/CRITICAL alerts to the CSV log."""
    level = get_alert_level(probability)
    alert = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "machine_id": machine_id,
        "failure_probability": round(float(probability), 4),
        "alert_level": level,
        "recommended_action": ACTIONS[level],
        "likely_causes": "; ".join(diagnose(reading)) if level != "NORMAL" else "-",
    }
    if log and level != "NORMAL":
        config.ensure_directories()
        new_file = not log_file.exists()
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(alert.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(alert)
    return alert


def format_alert(alert):
    icon = {"NORMAL": "[ OK ]", "WARNING": "[WARN]", "CRITICAL": "[CRIT]"}[alert["alert_level"]]
    text = (f"{icon} {alert['machine_id']} | failure risk {alert['failure_probability']:.1%} "
            f"| {alert['alert_level']} -> {alert['recommended_action']}")
    if alert["alert_level"] != "NORMAL":
        text += f"\n        cause: {alert['likely_causes']}"
    return text
