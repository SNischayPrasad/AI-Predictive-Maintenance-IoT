"""
model_training.py
-----------------
Trains several classifiers and keeps the best one.

Key ideas for beginners
  * Stratified split  - failures are rare (~3%), so we keep the same failure
                        ratio in train and test sets.
  * class_weight      - tells the model "missing a failure is expensive",
                        otherwise it learns to always say "no failure".
  * Pipeline          - scaler + model saved together, so prediction later
                        applies exactly the same scaling.
  * Cross-validation  - we choose the best model using the TRAIN set only,
                        so the test set stays an honest final exam.
"""

import time

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def split_data(df, feature_cols):
    X = df[feature_cols]
    y = df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    print(f"[SPLIT] Train: {len(X_train):,} rows ({y_train.sum()} failures) | "
          f"Test: {len(X_test):,} rows ({y_test.sum()} failures)")
    return X_train, X_test, y_train, y_test


def get_candidate_models():
    """Three models from simple -> powerful."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2,
            n_jobs=-1, random_state=config.RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=300, learning_rate=0.05,
            random_state=config.RANDOM_STATE),
    }


def train_models(X_train, y_train):
    """Cross-validate every model, fit it on the full train set, return results."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    trained, cv_rows = {}, []

    for name, model in get_candidate_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        start = time.time()
        f1_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        pipe.fit(X_train, y_train)
        elapsed = time.time() - start
        trained[name] = pipe
        cv_rows.append({"model": name, "cv_f1_mean": f1_scores.mean(),
                        "cv_f1_std": f1_scores.std(), "train_seconds": elapsed})
        print(f"[TRAIN] {name:<20} CV F1 = {f1_scores.mean():.3f} "
              f"(+/- {f1_scores.std():.3f})  [{elapsed:.1f}s]")

    cv_results = pd.DataFrame(cv_rows).sort_values("cv_f1_mean", ascending=False)
    best_name = cv_results.iloc[0]["model"]
    print(f"[TRAIN] Best model by cross-validated F1: {best_name}")
    return trained, cv_results, best_name


def save_model(pipeline, model_name, feature_cols, path=config.BEST_MODEL_FILE):
    """Save model + metadata in one file so predict.py knows the feature order."""
    config.ensure_directories()
    bundle = {"pipeline": pipeline, "model_name": model_name, "features": feature_cols}
    joblib.dump(bundle, path)
    print(f"[TRAIN] Saved best model -> {path}")


def load_model(path=config.BEST_MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python main.py train` first.")
    return joblib.load(path)
