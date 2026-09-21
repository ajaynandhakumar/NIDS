"""
model.py — ML Model Training & Persistence
===========================================
Trains a RandomForestClassifier on the synthetic (or real) network traffic
dataset, evaluates it, and persists the trained model to disk via joblib.
Provides a clean API: train_model(), load_model(), evaluate_model().
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Optional

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

# ------------------------------------------
# Configuration
# ------------------------------------------
MODEL_PATH      = "models/nids_model.pkl"
MODEL_VERSION   = "1.0"
TEST_SIZE       = 0.20     # 80/20 train-test split
RANDOM_STATE    = 42

# RandomForest hyper-parameters (tuned for small–medium NIDS datasets)
RF_PARAMS = {
    "n_estimators"      : 200,       # Number of trees
    "max_depth"         : 15,        # Prevent overfitting
    "min_samples_split" : 5,
    "min_samples_leaf"  : 2,
    "class_weight"      : "balanced",# Handle imbalanced attack/normal ratio
    "n_jobs"            : -1,        # Use all CPU cores
    "random_state"      : RANDOM_STATE,
}


# ------------------------------------------
# Training
# ------------------------------------------

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    save: bool = True,
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train a RandomForestClassifier and optionally save it to disk.

    Args:
        X    : Feature DataFrame (already preprocessed and normalised).
        y    : Binary label Series (0 = NORMAL, 1 = ATTACK).
        save : If True, persist the model to MODEL_PATH.

    Returns:
        Tuple[RandomForestClassifier, dict]:
            - Trained model instance.
            - Evaluation metrics dictionary.
    """
    print(f"[model] Training RandomForestClassifier on {len(X)} samples…")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )

    # -- Train --------------------------------------------------------------
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)

    # -- Evaluate -----------------------------------------------------------
    metrics = evaluate_model(clf, X_test, y_test, verbose=True)

    # -- 5-Fold Cross-Validation --------------------------------------------
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1", n_jobs=-1)
    metrics["cv_f1_mean"] = cv_scores.mean()
    metrics["cv_f1_std"]  = cv_scores.std()
    print(
        f"[model] Cross-Val F1: {cv_scores.mean():.4f} "
        f"(±{cv_scores.std():.4f})"
    )

    # -- Feature Importance -------------------------------------------------
    importance = dict(zip(X.columns, clf.feature_importances_))
    print("[model] Feature importances:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"          {feat:<15} {imp:.4f}  {bar}")

    # -- Save ---------------------------------------------------------------
    if save:
        _save_model(clf, X.columns.tolist())

    return clf, metrics


# ------------------------------------------
# Evaluation
# ------------------------------------------

def evaluate_model(
    clf    : RandomForestClassifier,
    X_test : pd.DataFrame,
    y_test : pd.Series,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a trained model and return a metrics dictionary.

    Args:
        clf     : Trained classifier.
        X_test  : Test features.
        y_test  : True labels.
        verbose : Print results to stdout.

    Returns:
        dict: accuracy, precision, recall, f1, roc_auc, confusion_matrix.
    """
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(
        y_test, y_pred,
        target_names=["NORMAL", "ATTACK"],
        output_dict=True,
    )

    if verbose:
        print("\n" + "-" * 52)
        print("  Model Evaluation Report")
        print("-" * 52)
        print(f"  Accuracy      : {acc * 100:.2f}%")
        print(f"  ROC-AUC Score : {roc_auc:.4f}")
        print(f"  Confusion Matrix:\n    TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
        print("\n" + classification_report(
            y_test, y_pred, target_names=["NORMAL", "ATTACK"]
        ))
        print("-" * 52 + "\n")

    return {
        "accuracy"        : acc,
        "roc_auc"         : roc_auc,
        "confusion_matrix": cm.tolist(),
        "report"          : report,
    }


# ------------------------------------------
# Persistence Helpers
# ------------------------------------------

def _save_model(clf: RandomForestClassifier, feature_cols: list) -> None:
    """
    Save the model and its metadata to disk.

    Args:
        clf          : Trained classifier.
        feature_cols : Ordered list of feature column names.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    payload = {
        "model"        : clf,
        "feature_cols" : feature_cols,
        "version"      : MODEL_VERSION,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"[model] [OK]  Model saved -> {MODEL_PATH}")


def load_model() -> Tuple[RandomForestClassifier, list]:
    """
    Load the trained model from disk.

    Returns:
        Tuple[RandomForestClassifier, list]:
            - Loaded model.
            - Feature column names (in training order).

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"[model] Model not found at '{MODEL_PATH}'. "
            "Run training first: python model.py"
        )

    payload = joblib.load(MODEL_PATH)
    clf          = payload["model"]
    feature_cols = payload["feature_cols"]
    version      = payload.get("version", "?")
    print(f"[model] Loaded model v{version} with {clf.n_estimators} trees.")
    return clf, feature_cols


def model_exists() -> bool:
    """Return True if a saved model file exists on disk."""
    return os.path.exists(MODEL_PATH)


# ------------------------------------------
# Convenience: train from scratch if needed
# ------------------------------------------

def ensure_trained() -> Tuple[RandomForestClassifier, list]:
    """
    Load the model if it exists, otherwise generate data, preprocess, and train.

    Returns:
        Tuple[RandomForestClassifier, list]: Model and feature column names.
    """
    if model_exists():
        return load_model()

    print("[model] No saved model found — training from scratch…")
    from dataset    import generate_dataset
    from preprocess import preprocess_dataset

    df          = generate_dataset(save=True)
    X, y        = preprocess_dataset(df, fit_scaler=True)
    clf, _      = train_model(X, y, save=True)
    return clf, list(X.columns)


# -- Quick standalone test ---------------------------------------------------
if __name__ == "__main__":
    clf, cols = ensure_trained()
    print(f"\nModel type : {type(clf).__name__}")
    print(f"Features   : {cols}")
    print(f"Classes    : {clf.classes_}")
