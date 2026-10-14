"""
config.py
---------
Central place for every path and setting used in the project.
Change values here instead of hunting through the code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folder paths (all relative to the project root, so it works on any machine)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
IMAGES_DIR = ROOT_DIR / "images"

# Real dataset file name (UCI AI4I 2020). If you download it and place it in
# data/raw/, the project uses it automatically. Otherwise the simulator creates
# a synthetic dataset with the same structure.
REAL_DATASET_FILE = DATA_RAW_DIR / "ai4i2020.csv"
SIMULATED_DATASET_FILE = DATA_RAW_DIR / "simulated_sensor_data.csv"
PROCESSED_DATASET_FILE = DATA_PROCESSED_DIR / "processed_sensor_data.csv"

BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = OUTPUTS_DIR / "model_metrics.csv"
PREDICTIONS_FILE = OUTPUTS_DIR / "test_predictions.csv"
ALERTS_LOG_FILE = OUTPUTS_DIR / "alerts_log.csv"
STREAM_LOG_FILE = OUTPUTS_DIR / "stream_simulation_log.csv"

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
N_SAMPLES = 10_000        # number of sensor records to simulate
RANDOM_STATE = 42         # fixed seed -> same results every run (reproducible)

# ---------------------------------------------------------------------------
# Column names (match the AI4I 2020 dataset exactly)
# ---------------------------------------------------------------------------
COL_TYPE = "Type"
COL_AIR_TEMP = "Air temperature [K]"
COL_PROCESS_TEMP = "Process temperature [K]"
COL_RPM = "Rotational speed [rpm]"
COL_TORQUE = "Torque [Nm]"
COL_TOOL_WEAR = "Tool wear [min]"
TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Columns that must NEVER be used as model inputs:
#  - IDs carry no physical meaning
#  - failure-mode flags are only known AFTER the failure (data leakage)
DROP_COLUMNS = ["UDI", "Product ID"] + FAILURE_MODES

# ---------------------------------------------------------------------------
# Model / alert settings
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
WARNING_THRESHOLD = 0.30   # failure probability >= 30% -> WARNING
CRITICAL_THRESHOLD = 0.60  # failure probability >= 60% -> CRITICAL


def ensure_directories():
    """Create every output folder if it does not exist yet."""
    for folder in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
