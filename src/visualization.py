"""
visualization.py
----------------
Creates every chart for the README / LinkedIn post / interview demo.
All images are saved to the images/ folder with numbered names so they
appear in a logical order.
"""

import matplotlib

matplotlib.use("Agg")  # save images without opening windows (works everywhere)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src import config

sns.set_theme(style="whitegrid", palette="deep")
COLORS = {0: "#2a9d8f", 1: "#e63946"}   # healthy = teal, failure = red


def _save(fig, name):
    config.ensure_directories()
    path = config.IMAGES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[VIZ] Saved {path.name}")


def plot_sensor_distributions(df):
    sensors = [config.COL_AIR_TEMP, config.COL_PROCESS_TEMP, config.COL_RPM,
               config.COL_TORQUE, config.COL_TOOL_WEAR]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, sensors):
        for label, name in [(0, "No Failure"), (1, "Failure")]:
            sns.kdeplot(df.loc[df[config.TARGET] == label, col], ax=ax, fill=True,
                        color=COLORS[label], label=name, common_norm=False, warn_singular=False, cut=0)
        ax.set_title(col)
        ax.legend()
    counts = df[config.TARGET].value_counts().sort_index()
    axes.flat[5].bar(["No Failure", "Failure"], counts.values, color=[COLORS[0], COLORS[1]])
    axes.flat[5].set_title("Class balance (failures are rare!)")
    for i, v in enumerate(counts.values):
        axes.flat[5].text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.suptitle("IoT Sensor Readings: Healthy vs Failing Machines", fontsize=15, weight="bold")
    _save(fig, "01_sensor_distributions.png")


def plot_correlation_heatmap(df_features):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_features.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, weight="bold")
    _save(fig, "02_correlation_heatmap.png")


def plot_failure_modes(raw_df):
    counts = raw_df[config.FAILURE_MODES].sum().sort_values(ascending=False)
    names = {"TWF": "Tool Wear", "HDF": "Heat Dissipation", "PWF": "Power",
             "OSF": "Overstrain", "RNF": "Random"}
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([f"{m}\n{names[m]}" for m in counts.index], counts.values, color="#e76f51")
    ax.bar_label(bars)
    ax.set_ylabel("Number of failures")
    ax.set_title("Failure Root Causes in the Dataset", fontsize=14, weight="bold")
    _save(fig, "03_failure_modes.png")


def plot_model_comparison(metrics):
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    data = metrics.melt(id_vars="model", value_vars=cols, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=data, x="metric", y="score", hue="model", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)
    ax.set_ylim(0, 1.22)
    ax.set_title("Model Comparison on Unseen Test Data", fontsize=14, weight="bold")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    _save(fig, "04_model_comparison.png")


def plot_confusion_matrix(name, pipeline, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, cmap="Blues",
                                          display_labels=["No Failure", "Failure"], colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}", fontsize=13, weight="bold")
    _save(fig, "05_confusion_matrix.png")


def plot_roc_curves(trained_models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in trained_models.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_title("ROC Curves", fontsize=14, weight="bold")
    ax.legend(loc="lower right")
    _save(fig, "06_roc_curves.png")


def plot_feature_importance(name, pipeline, X_test, y_test):
    """Permutation importance works for ANY model: shuffle a feature, see how much F1 drops."""
    result = permutation_importance(pipeline, X_test, y_test, scoring="f1", n_repeats=10,
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    imp.plot.barh(ax=ax, color="#264653")
    ax.set_xlabel("Drop in F1 score when feature is shuffled")
    ax.set_title(f"What Drives Failure Predictions? ({name})", fontsize=13, weight="bold")
    _save(fig, "07_feature_importance.png")
    return imp


def plot_probability_distribution(predictions):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, name in [(0, "Actually healthy"), (1, "Actually failed")]:
        subset = predictions.loc[predictions["actual"] == label, "failure_probability"]
        ax.hist(subset, bins=40, alpha=0.7, color=COLORS[label], label=name, log=True)
    ax.axvline(config.WARNING_THRESHOLD, color="orange", ls="--", label="WARNING threshold")
    ax.axvline(config.CRITICAL_THRESHOLD, color="red", ls="--", label="CRITICAL threshold")
    ax.set_xlabel("Predicted failure probability")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Predicted Failure Risk on Test Machines", fontsize=14, weight="bold")
    ax.legend()
    _save(fig, "08_failure_probability_distribution.png")


def plot_stream_simulation(log_df):
    """Live-monitoring style dashboard for the virtual machine simulation."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    t = log_df["cycle"]

    axes[0].plot(t, log_df["Tool wear [min]"], color="#6d597a")
    axes[0].set_ylabel("Tool wear\n[min]")
    axes[1].plot(t, log_df["Air temperature [K]"], label="Air temp", color="#457b9d")
    axes[1].plot(t, log_df["Process temperature [K]"], label="Process temp", color="#e76f51")
    axes[1].set_ylabel("Temperature\n[K]")
    axes[1].legend(loc="upper left")
    axes[2].plot(t, log_df["Torque [Nm]"], color="#2a9d8f")
    axes[2].set_ylabel("Torque\n[Nm]")

    ax = axes[3]
    ax.plot(t, log_df["failure_probability"], color="black", lw=1.2)
    ax.axhspan(config.CRITICAL_THRESHOLD, 1.0, color="red", alpha=0.12, label="CRITICAL zone")
    ax.axhspan(config.WARNING_THRESHOLD, config.CRITICAL_THRESHOLD, color="orange", alpha=0.12,
               label="WARNING zone")
    crit = log_df[log_df["alert_level"] == "CRITICAL"]
    ax.scatter(crit["cycle"], crit["failure_probability"], color="red", s=18, zorder=3,
               label="Critical alert")
    maint = log_df[log_df["event"] == "MAINTENANCE"]
    for c in maint["cycle"]:
        for a in axes:
            a.axvline(c, color="green", ls=":", lw=1.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Failure\nprobability")
    ax.set_xlabel("Machine cycle (time)")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Virtual IoT Machine - Real-Time Predictive Maintenance Monitor\n"
                 "(green dotted line = maintenance performed after a CRITICAL alert)",
                 fontsize=14, weight="bold")
    _save(fig, "09_realtime_monitoring_simulation.png")
