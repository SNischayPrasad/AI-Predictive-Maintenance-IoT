# 📘 Complete Project Guide: AI-Powered Predictive Maintenance for IoT

This guide walks through the project from zero to GitHub, step by step. Sections A–M follow the order you asked for.

---

## A. Project Explanation

### Simple language
Think of a car. You can:
- **Wait until it breaks down** on the highway (reactive: expensive and dangerous)
- **Change the oil every 5,000 km** whether it needs it or not (preventive: wasteful)
- **Have the car tell you** "the engine is overheating, oil is low, service in 3 days" (**predictive**)

Predictive maintenance gives machines that third option. Sensors watch the machine, an AI model learns what "about to fail" looks like, and it warns you **before** the breakdown.

### Technical language
Predictive Maintenance (PdM) is a condition-based maintenance strategy. Multivariate time-series telemetry from IoT sensors (temperature, vibration, pressure, current, torque, RPM) is ingested, cleaned and transformed into health indicators. A supervised model (classification for "failure within horizon", or regression for Remaining Useful Life) estimates failure probability. Its output is thresholded into alerts that feed a CMMS (Computerized Maintenance Management System) to schedule work orders.

### Problem it solves
| Problem | Impact | How PdM helps |
|---|---|---|
| Unplanned downtime | A car assembly line can lose ~$20k+ per minute | Fix before failure, during planned stops |
| Over-maintenance | Healthy parts thrown away | Replace parts only when needed |
| Safety incidents | Exploding turbines, aircraft faults | Early warning |
| Spare-parts inventory | Capital locked in warehouses | Order parts just in time |

### Industry usage
- **Manufacturing plants / factories**: CNC tool wear, spindle bearings, conveyor motors. Siemens and Bosch use vibration and current sensors.
- **Power plants**: turbine blade vibration, generator winding temperature, transformer oil analysis. GE uses Predix "digital twins".
- **Automotive**: welding robots on assembly lines, plus connected cars that predict battery or brake wear.
- **Aviation**: jet engines stream thousands of parameters per flight. Rolls-Royce "Power-by-the-Hour" and NASA's C-MAPSS turbofan dataset are both built around Remaining Useful Life.

Companies typically report **30–50% less downtime** and **10–40% lower maintenance cost**. Treat these as industry survey estimates, not guarantees.

### Full workflow
```
1. Sensor data collection   -> temperature, vibration, torque, RPM, pressure sent every few seconds via MQTT/OPC-UA
2. Preprocessing            -> remove duplicates, fix sensor glitches, fill dropped packets
3. Feature engineering      -> power = torque x speed, temp difference, strain, rolling averages
4. Model training           -> learn patterns from history where failures are labelled
5. Prediction               -> new reading -> failure probability -> Failure / No Failure
6. Alert generation         -> NORMAL / WARNING / CRITICAL + likely cause + recommended action
7. Visualization            -> dashboards for engineers and managers
```

---

## B. Tech Stack Options

| | Option A: Easiest | Option B: Intermediate ✅ | Option C: Advanced |
|---|---|---|---|
| Tools | Pandas, Scikit-learn, Matplotlib (single notebook) | Pandas, NumPy, Scikit-learn, Seaborn, Joblib, Pytest, **modular `src/` + virtual IoT simulator** | TensorFlow/PyTorch, Streamlit, FastAPI, MQTT, Docker |
| Dataset | AI4I 2020 tabular CSV | AI4I-style sensor data (simulated or real) + streaming simulation | NASA C-MAPSS turbofan time-series (multi-sensor, run-to-failure) |
| Models | Logistic Regression, Decision Tree | Logistic Regression, Random Forest, Gradient Boosting + cross-validation | LSTM / 1D-CNN for Remaining Useful Life, XGBoost |
| Difficulty | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| GPU | No | No | Recommended for LSTM |
| Time | 1–2 days | 5–7 days | 3–4 weeks |
| Output | Accuracy and a confusion matrix | Trained model, metrics, 9 charts, alert system, live simulation, tests | RUL predictions, live web dashboard, API |
| GitHub proof | Weak (looks like a tutorial) | **Strong** (looks like an engineering project) | Very strong, but hard to finish |

## C. Selected Approach: **Option B**

Why Option B:
- Runs on any laptop in about 1–2 minutes, with no GPU and no downloads
- Uses the libraries recruiters expect (Pandas, NumPy, Scikit-learn)
- Modular code, tests and a CLI read as **software engineering**, not just a notebook
- The virtual IoT stream plus the alert system makes the work demo-able in interviews
- Option C can be added later as "v2", which gives you a natural roadmap

---

## D. Architecture

```
┌─────────────────────────────── DATA LAYER ───────────────────────────────┐
│  data_simulator.py  ──►  data/raw/*.csv  ◄── (optional) real ai4i2020.csv  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    ▼
┌─────────────────────────── PROCESSING LAYER ─────────────────────────────┐
│  data_loader.py ─► preprocessing.py ─► feature_engineering.py            │
│  (load)            (dedupe, glitches,   (Power, Temp diff, Strain,       │
│                     impute, drop leaks)  Type encoding)                  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    ▼
┌──────────────────────────── ML LAYER ────────────────────────────────────┐
│  model_training.py: stratified split ─► 5-fold CV on 3 models ─► best    │
│  evaluation.py: accuracy, precision, recall, F1, ROC-AUC on test set     │
│  models/best_model.joblib (scaler + model + feature list)                │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    ▼
┌────────────────────────── SERVING / ACTION LAYER ────────────────────────┐
│  predict.py ─► alert_system.py ─► outputs/alerts_log.csv                 │
│  iot_stream_simulator.py: live packets ─► predict ─► alert ─► maintenance│
└───────────────────────────────────┬──────────────────────────────────────┘
                                    ▼
┌────────────────────────── PRESENTATION LAYER ────────────────────────────┐
│  visualization.py ─► images/01..09 *.png  ─► README dashboard            │
└──────────────────────────────────────────────────────────────────────────┘
```

**Modules**
| Module | Input | Output | Role |
|---|---|---|---|
| `config.py` | – | constants | Single source of truth for paths and thresholds |
| `data_simulator.py` | seed, n | DataFrame | Virtual factory: generates sensor readings and failure labels |
| `data_loader.py` | CSV | DataFrame | Picks the real or simulated dataset |
| `preprocessing.py` | raw df | clean df | Duplicates, impossible values, missing values, leakage removal |
| `feature_engineering.py` | clean df | feature df | Domain features |
| `model_training.py` | features | trained pipelines | CV model selection, saves the best one |
| `evaluation.py` | model + test | metrics CSV | Honest performance on unseen data |
| `predict.py` | new reading | probability | Inference |
| `alert_system.py` | probability | alert | Business action + root cause |
| `iot_stream_simulator.py` | model | log + chart | Real-time monitoring demo |
| `visualization.py` | data/metrics | PNGs | Charts |

**Data flow:** `sensor reading (6 values) → clean → 9 features → model → P(failure) → alert level → log + dashboard`

---

## E. Folder Structure

```
AI-Predictive-Maintenance-IoT/
├── data/raw/          # Input data. Simulated CSV committed (small, reproducible). Real UCI CSV ignored by git.
├── data/processed/    # Engineered features. Regenerated every run, so ignored by git.
├── notebooks/         # Jupyter EDA: storytelling + exploration (recruiters like to see this)
├── src/               # Production-style Python modules (the "real" code)
├── models/            # Saved trained model (.joblib) so anyone can predict without retraining
├── outputs/           # Metrics CSV, predictions CSV, stream log, full run log
├── images/            # Charts embedded in README (numbered = presentation order)
├── docs/              # This guide
├── tests/             # Pytest unit tests
├── main.py            # CLI entry point
├── requirements.txt   # Dependencies
├── pytest.ini         # Test config
├── .gitignore         # Files git should skip
└── README.md          # Project landing page
```

---

## F. Installation & Environment Setup

**Python version:** 3.10, 3.11, 3.12 or 3.13 (tested on 3.13).

### Windows (PowerShell)
```powershell
python --version
cd AI-Predictive-Maintenance-IoT
python -m venv venv
venv\Scripts\activate
# If you see "running scripts is disabled", run once:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Mac / Linux
```bash
python3 --version
cd AI-Predictive-Maintenance-IoT
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Verify
```bash
python -c "import pandas, sklearn, seaborn; print('Setup OK')"
pytest
```

**requirements.txt**
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
matplotlib>=3.7
seaborn>=0.13
pytest>=7.4
jupyter>=1.0
```

---

## G. Code Walkthrough

All code is in `src/` and `main.py`, fully commented. What each file does and why:

| File | Key concept to explain in an interview |
|---|---|
| `src/config.py` | No hard-coded paths. `Path(__file__)` makes the project portable across machines. |
| `src/data_simulator.py` | Physics rules: power `P = τ·ω`, heat dissipation `ΔT < 8.6 K && rpm < 1380`, overstrain `wear × torque > limit`. Random walk for temperature drift. |
| `src/preprocessing.py` | **Data leakage**: TWF/HDF/PWF/OSF/RNF columns are only known after a failure, so they are dropped. Median imputation is robust to outliers. |
| `src/feature_engineering.py` | Domain knowledge beats raw data. "Power" is the #1 feature in importance. |
| `src/model_training.py` | Stratified split, `class_weight="balanced"`, `Pipeline(scaler + model)`, 5-fold CV for selection. |
| `src/evaluation.py` | Why accuracy is misleading at 96% healthy; recall = caught failures. |
| `src/predict.py` | Saved bundle stores the feature order, which prevents train/serve skew. |
| `src/alert_system.py` | Thresholds turn probability into action, plus rule-based root cause. |
| `src/iot_stream_simulator.py` | Closed loop: sense → predict → alert → maintain. |
| `src/visualization.py` | Permutation importance works for any model. |
| `tests/test_pipeline.py` | Simulator sanity, leakage removal, glitch fixing, physics formulas, alert levels. |

---

## H. Virtual Simulation

### How the dataset simulates real IoT data
Each row represents one machining cycle reported by a machine's IoT gateway:
- **Air temperature** follows a random walk around 300 K (daily or seasonal drift, not pure noise)
- **Process temperature** = air + ~10 K (the machine heats up while cutting)
- **Torque and RPM** are negatively linked (heavier load slows the spindle), with occasional **load spikes** (jammed workpiece)
- **Tool wear** grows each cycle (faster for high-quality products) and resets when the tool is replaced

### How failure is represented
`Machine failure = 1` if any physical rule triggers:

| Mode | Rule | Real-world meaning |
|---|---|---|
| TWF | tool wear 200–240 min (random chance) | Cutting tool snaps |
| HDF | ΔT < 8.6 K and rpm < 1380 | Cooling fan or coolant problem |
| PWF | power < 3500 W or > 9000 W | Motor underload or overload |
| OSF | wear × torque > 11k/12k/13k (L/M/H) | Worn tool pushed too hard |
| RNF | 0.1% random | Unpredictable failures (no model can catch these) |

### How the model predicts breakdown
The model never sees these rules. It only sees sensor values and labels, and it **learns** the failure boundaries from examples. The feature-importance chart shows it rediscovered that Power, Torque and Strain matter most.

### Step-by-step simulation
1. **Step 1: Generate the fleet.** `python main.py simulate` creates 10,000 cycles in `data/raw/simulated_sensor_data.csv`.
2. **Step 2: Inspect.** Open the CSV or the notebook. About 3.9% of cycles are failures.
3. **Step 3: Train.** `python main.py train` cleans the data, engineers features, cross-validates 3 models and saves the best one.
4. **Step 4: Test on new machines.** The demo scenarios print alerts for a healthy machine, a worn tool, a cooling fault and an overload.
5. **Step 5: Go live.** `python main.py stream --delay 0.2` brings virtual machine **VM-07** online and streams 320 sensor packets:
   - cycles 70–110: cooling fan degrades → WARNING → CRITICAL → maintenance
   - tool wears past ~210 min → CRITICAL → tool replaced
   - cycles 250–265: motor overload → CRITICAL within 1 cycle → maintenance
6. **Step 6: Review.** Look at `images/09_realtime_monitoring_simulation.png` and `outputs/alerts_log.csv`.
7. **Step 7: Try your own reading.** `python main.py predict --type H --torque 70 --rpm 1250 --wear 60`

### Proof to capture
- Screenshot of the terminal during `stream` (WARN/CRIT lines)
- A **screen recording or GIF** (10–20 s) of `python main.py stream --delay 0.2`, which makes a strong LinkedIn post
- `images/09_realtime_monitoring_simulation.png`
- `outputs/alerts_log.csv` opened in Excel

---

## I. Execution

| Task | Command | Output |
|---|---|---|
| Everything | `python main.py` | All CSVs, model, 9 charts, logs |
| Train only | `python main.py train` | `models/best_model.joblib`, `outputs/model_metrics.csv` |
| Test a prediction | `python main.py predict --type L --air 300 --process 310.5 --rpm 1380 --torque 58 --wear 215` | Alert line |
| Visualize | open `images/` | 9 PNG charts |
| Live demo | `python main.py stream --delay 0.2` | Live alerts + chart 09 |
| Unit tests | `pytest -v` | 9 passed |
| Notebook | `jupyter notebook notebooks/01_exploratory_data_analysis.ipynb` | EDA |

### Sample output (actual run)
```
[TRAIN] Logistic Regression  CV F1 = 0.405 (+/- 0.011)
[TRAIN] Random Forest        CV F1 = 0.871 (+/- 0.044)
[TRAIN] Gradient Boosting    CV F1 = 0.869 (+/- 0.027)
[TRAIN] Best model by cross-validated F1: Random Forest

[EVAL] Test-set results (unseen data):
              model  accuracy  precision  recall    f1  roc_auc
      Random Forest     0.992      0.930   0.857 0.892    0.982
  Gradient Boosting     0.990      0.890   0.844 0.867    0.979
Logistic Regression     0.899      0.259   0.870 0.399    0.956

[PREDICT] Model caught 66 of 77 real failures in the test set.

[ OK ] MACHINE-01 (healthy) | failure risk 0.0% | NORMAL -> Continue normal operation
[CRIT] MACHINE-02 (worn tool, high load) | failure risk 96.0% | CRITICAL -> STOP machine ...
[CRIT] MACHINE-03 (cooling problem) | failure risk 62.3% | CRITICAL -> STOP machine ...
[CRIT] MACHINE-04 (power overload) | failure risk 96.5% | CRITICAL -> STOP machine ...

[STREAM] Maintenance actions triggered: 3
```
Your exact numbers can differ slightly by library version. The full log is in `outputs/run_log.txt`.

### Common mistakes and fixes
| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root folder, not from inside `src/` |
| `No trained model at ...` | Run `python main.py train` before `predict` or `stream` |
| `'pip' is not recognized` | Use `python -m pip install -r requirements.txt` |
| venv activation blocked on Windows | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Charts don't pop up | By design: they are saved to `images/` |

---

## Phases 1–10 (summary)

| Phase | What | Why | Expected output | Common mistake | Verify |
|---|---|---|---|---|---|
| 1 Setup | venv + requirements | Isolated, reproducible env | `Setup OK` | Installing globally | `pip list` inside venv |
| 2 Load | `data_loader.py` | Get sensor data | 10,000 × 14 table | Wrong working directory | Shape printed |
| 3 Clean | `preprocessing.py` | Garbage in = garbage out | Leakage columns dropped | Keeping TWF/HDF columns (fake 100% accuracy!) | Columns list printed |
| 4 Features | `feature_engineering.py` | Encode physics | 9 inputs | Forgetting to encode Type | `processed_sensor_data.csv` |
| 5 Model | `model_training.py` | Learn failure patterns | 3 models, CV F1 | Not handling imbalance | CV scores printed |
| 6 Evaluate | `evaluation.py` | Honest performance | metrics table | Judging by accuracy only | Recall > 0.8 |
| 7 Predict | `predict.py` | Use the model | Probabilities + alerts | Feature order mismatch | Demo scenarios behave sensibly |
| 8 Visualize | `visualization.py` | Communicate results | 9 PNGs | Unlabelled charts | Open `images/` |
| 9 GitHub | Section J | Proof of work | Public repo | Committing venv/ | Repo looks clean |
| 10 Final | README + demo | Showcase | Polished repo + post | No screenshots | Ask a friend to clone and run |

---

## J. GitHub Upload Strategy

**Repo name:** `AI-Predictive-Maintenance-IoT`
**Description:** `End-to-end AI predictive maintenance system: simulated IoT sensor data, ML failure prediction (Random Forest, F1 0.89), real-time alerting and virtual machine monitoring. Python · Scikit-learn · Pandas`
**Topics/tags:** `predictive-maintenance` `iot` `machine-learning` `industry-4-0` `scikit-learn` `python` `data-science` `anomaly-detection` `industrial-ai` `random-forest` `time-series` `manufacturing`

### Steps
1. github.com → **New repository** → name it as above → Public → **don't** tick "Add README" (you already have one).
2. In the project folder:
```bash
git init
git add .
git commit -m "feat: initial predictive maintenance pipeline"
git branch -M main
git remote add origin https://github.com/SNischayPrasad/AI-Predictive-Maintenance-IoT.git
git push -u origin main
```
3. On the repo page → ⚙️ next to **About** → paste the description and topics.
4. Replace `SNischayPrasad` and `<your-profile>` in README.md, then commit and push.
5. Pin the repo on your GitHub profile (Profile → Customize your pins).
6. Optional: add a **GitHub Actions** workflow that runs `pytest` on every push, for a green ✅ badge.

### Commit message style (Conventional Commits)
`feat:` new feature · `fix:` bug fix · `docs:` documentation · `test:` tests · `refactor:` restructure · `chore:` setup

### Making it professional
- Charts in README (already done) · badges · results table · honest-limitations note
- Clean `.gitignore` (no `venv/`, no `__pycache__`)
- A `LICENSE` (MIT included)
- A 30–60 s demo GIF at the top of the README (record the `stream` command with ScreenToGif or OBS)

---

## K. README
`README.md` at the project root is complete and copy-paste ready. Only replace the `SNischayPrasad` and `<your-profile>` placeholders.

---

## L. 7-Day Commit Plan (GitHub proof)

Spread the work over real days so your contribution graph shows consistent activity. Never fake commit dates.

| Day | Work | Files | Commit message | Proof to capture |
|---|---|---|---|---|
| 1 Setup | Repo, venv, structure, config | `requirements.txt`, `.gitignore`, `src/config.py`, folders | `chore: project structure and environment setup` | Screenshot of the empty repo + folder tree |
| 2 Dataset | Simulator + loader + EDA notebook | `data_simulator.py`, `data_loader.py`, `notebooks/`, `data/raw/` | `feat: virtual IoT sensor simulator and data loader` + `docs: EDA notebook` | `01_sensor_distributions.png`, dataset preview |
| 3 Preprocessing | Cleaning + features + tests | `preprocessing.py`, `feature_engineering.py`, `tests/` | `feat: data cleaning and physics-based feature engineering` + `test: unit tests for pipeline` | `pytest` passing screenshot |
| 4 Model | Training with CV | `model_training.py`, `models/` | `feat: train and cross-validate LR, RF, GB models` | Training log screenshot |
| 5 Evaluation | Metrics + prediction + alerts | `evaluation.py`, `predict.py`, `alert_system.py` | `feat: evaluation metrics, prediction and alert system` | Confusion matrix, metrics table |
| 6 Visualization | Charts + live stream | `visualization.py`, `iot_stream_simulator.py`, `main.py`, `images/` | `feat: real-time IoT monitoring simulation and dashboards` | Chart 09, GIF of the live stream |
| 7 Upload | README, guide, polish | `README.md`, `docs/`, `LICENSE` | `docs: complete README with results and architecture` | Repo homepage screenshot, LinkedIn post |

---

## M. Proof / Demo Assets Checklist

| # | Asset | File name | Where | Used in |
|---|---|---|---|---|
| 1 | Dataset preview | `images/proof_dataset_preview.png` (screenshot of `head()`) | images/ | README / LinkedIn |
| 2 | Preprocessing output | `images/proof_cleaning_log.png` (STEP 3 terminal) | images/ | Blog / post |
| 3 | Training logs | `images/proof_training_log.png` + `outputs/run_log.txt` | images/, outputs/ | README |
| 4 | Prediction output | `images/proof_predictions.png` (demo scenarios) | images/ | README Usage |
| 5 | Failure detection graph | `images/09_realtime_monitoring_simulation.png` ✅ auto | images/ | README hero image |
| 6 | Confusion matrix | `images/05_confusion_matrix.png` ✅ auto | images/ | README Results |
| 7 | Model comparison / ROC | `images/04_model_comparison.png`, `06_roc_curves.png` ✅ auto | images/ | README Results |
| 8 | Feature importance | `images/07_feature_importance.png` ✅ auto | images/ | Interview talking point |
| 9 | Live demo GIF | `images/demo_stream.gif` | images/ | Top of README, LinkedIn |
| 10 | Tests passing | `images/proof_tests_passing.png` | images/ | README badge proof |
| 11 | GitHub repo preview | `images/proof_github_repo.png` | images/ (or LinkedIn only) | Portfolio / resume |

**Embed in README:** `![Confusion Matrix](images/05_confusion_matrix.png)`

### Resume bullet (copy-paste)
> **AI-Powered Predictive Maintenance System for IoT** | Python, Scikit-learn, Pandas
> • Built an end-to-end ML pipeline on 10,000 simulated IoT sensor records (AI4I 2020 schema) to predict industrial machine failures; Random Forest achieved **0.89 F1 / 0.98 ROC-AUC** and caught **86% of failures** with under 0.3% false alarms.
> • Engineered physics-based features (mechanical power, heat dissipation, tool strain) and handled 25:1 class imbalance with class weighting and stratified cross-validation.
> • Developed a real-time virtual machine monitor with a 3-level alert system and root-cause diagnosis that triggered maintenance before breakdown in all 3 injected fault scenarios.

### Interview questions to prepare
1. Why not use accuracy? (imbalance: 96% healthy)
2. What is data leakage and where could it happen here? (failure-mode columns)
3. Why did Logistic Regression underperform? (non-linear, two-sided failure boundaries)
4. Precision vs recall trade-off in maintenance: what costs more, a missed failure or a false alarm?
5. How would you deploy this in a real factory? (MQTT → edge/cloud inference → CMMS work orders)
6. How would you extend it to Remaining Useful Life? (regression or LSTM on C-MAPSS)
