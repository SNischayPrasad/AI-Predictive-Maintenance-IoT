"""
main.py - AI-Powered Predictive Maintenance System for IoT Devices
==================================================================

Usage
    python main.py                 # full pipeline: simulate -> train -> evaluate -> visualize -> demo
    python main.py simulate        # only generate the virtual IoT dataset
    python main.py train           # load/clean/engineer data, train + evaluate models, make charts
    python main.py predict --type L --air 300 --process 310.5 --rpm 1380 --torque 58 --wear 215
    python main.py stream          # real-time virtual machine monitoring simulation
    python main.py stream --delay 0.2   # slower, "live" looking output (good for screen recording)
"""

import argparse

import pandas as pd

from src import config
from src.data_loader import describe_dataset, load_dataset
from src.data_simulator import save_simulated_data, simulate_sensor_data
from src.evaluation import evaluate_all, print_report
from src.feature_engineering import add_features, get_feature_columns
from src.iot_stream_simulator import run_stream_simulation
from src.model_training import save_model, split_data, train_models
from src.predict import demo_predictions, predict_single
from src.preprocessing import clean_data
from src import visualization as viz


def banner(text):
    print("\n" + "=" * 70 + f"\n  {text}\n" + "=" * 70)


def run_simulate():
    banner("STEP 1 - VIRTUAL IoT SENSOR SIMULATION")
    df = simulate_sensor_data()
    save_simulated_data(df)
    return df


def run_training():
    banner("STEP 2 - LOAD DATASET")
    raw_df, _ = load_dataset()
    describe_dataset(raw_df)

    banner("STEP 3 - DATA CLEANING")
    clean_df = clean_data(raw_df)

    banner("STEP 4 - FEATURE ENGINEERING")
    feat_df = add_features(clean_df)
    feat_df.to_csv(config.PROCESSED_DATASET_FILE, index=False)
    feature_cols = get_feature_columns(feat_df)
    print(f"[FEATURES] {len(feature_cols)} model inputs: {feature_cols}")
    print(f"[FEATURES] Processed data saved -> {config.PROCESSED_DATASET_FILE}")

    banner("STEP 5 - MODEL TRAINING (5-fold cross-validation)")
    X_train, X_test, y_train, y_test = split_data(feat_df, feature_cols)
    trained, cv_results, best_name = train_models(X_train, y_train)
    cv_results.to_csv(config.OUTPUTS_DIR / "cross_validation_results.csv", index=False)
    best_pipe = trained[best_name]
    save_model(best_pipe, best_name, feature_cols)

    banner("STEP 6 - EVALUATION ON UNSEEN TEST DATA")
    metrics = evaluate_all(trained, X_test, y_test)
    print_report(best_name, best_pipe, X_test, y_test)

    banner("STEP 7 - FAILURE PREDICTION ON TEST MACHINES")
    preds = X_test.copy()
    preds["actual"] = y_test.values
    preds["failure_probability"] = best_pipe.predict_proba(X_test)[:, 1].round(4)
    preds["predicted"] = best_pipe.predict(X_test)
    preds.to_csv(config.PREDICTIONS_FILE, index=False)
    caught = ((preds["actual"] == 1) & (preds["predicted"] == 1)).sum()
    print(f"[PREDICT] Model caught {caught} of {preds['actual'].sum()} real failures "
          f"in the test set.")
    print(f"[PREDICT] Highest-risk test machines:")
    print(preds.sort_values("failure_probability", ascending=False)
          .head(5)[["Tool wear [min]", "Torque [Nm]", "Power [W]", "Temp difference [K]",
                    "failure_probability", "actual"]].round(2).to_string())
    print(f"[PREDICT] All predictions saved -> {config.PREDICTIONS_FILE}")

    banner("STEP 8 - VISUALIZATION")
    viz.plot_sensor_distributions(clean_df)
    viz.plot_correlation_heatmap(feat_df)
    if set(config.FAILURE_MODES).issubset(raw_df.columns):
        viz.plot_failure_modes(raw_df)
    viz.plot_model_comparison(metrics)
    viz.plot_confusion_matrix(best_name, best_pipe, X_test, y_test)
    viz.plot_roc_curves(trained, X_test, y_test)
    importance = viz.plot_feature_importance(best_name, best_pipe, X_test, y_test)
    print("[VIZ] Feature importance (top 3): "
          + ", ".join(importance.sort_values(ascending=False).head(3).index))
    viz.plot_probability_distribution(preds)

    demo_predictions()
    return metrics


def run_stream(delay):
    banner("STEP 9 - REAL-TIME IoT MONITORING SIMULATION")
    log_df = run_stream_simulation(delay=delay)
    viz.plot_stream_simulation(log_df)


def main():
    parser = argparse.ArgumentParser(description="AI-Powered Predictive Maintenance for IoT")
    parser.add_argument("command", nargs="?", default="all",
                        choices=["all", "simulate", "train", "predict", "stream"])
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between stream prints")
    parser.add_argument("--type", default="M", choices=["L", "M", "H"], help="product quality")
    parser.add_argument("--air", type=float, default=300.0, help="air temperature [K]")
    parser.add_argument("--process", type=float, default=310.0, help="process temperature [K]")
    parser.add_argument("--rpm", type=float, default=1500, help="rotational speed [rpm]")
    parser.add_argument("--torque", type=float, default=40.0, help="torque [Nm]")
    parser.add_argument("--wear", type=float, default=100, help="tool wear [min]")
    args = parser.parse_args()

    config.ensure_directories()

    if args.command == "simulate":
        run_simulate()
    elif args.command == "train":
        run_training()
    elif args.command == "predict":
        banner("SINGLE PREDICTION")
        predict_single("CUSTOM-MACHINE", product_type=args.type, air_temp=args.air,
                       process_temp=args.process, rpm=args.rpm, torque=args.torque,
                       tool_wear=args.wear)
    elif args.command == "stream":
        run_stream(args.delay)
    else:
        # Fresh alert log for a clean full run
        if config.ALERTS_LOG_FILE.exists():
            config.ALERTS_LOG_FILE.unlink()
        if not config.REAL_DATASET_FILE.exists():
            run_simulate()
        run_training()
        run_stream(args.delay)
        banner("DONE - check the images/, outputs/ and models/ folders")


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()
