"""AI training and inference pipeline for heart disease prediction.

This module is intentionally UI-agnostic so it can be reused by Streamlit,
notebooks, tests, or APIs.
"""

# Postpone annotation evaluation so forward references remain lightweight.
from __future__ import annotations

# Dataclass keeps model metadata grouped and explicit.
from dataclasses import dataclass

# Path helps build a reliable local dataset path.
from pathlib import Path

# Type aliases used throughout function signatures and return values.
from typing import Any, Dict, List, Tuple

# NumPy powers numeric threshold sweeps and CV summary stats.
import numpy as np

# Pandas handles tabular data loading and inference frame creation.
import pandas as pd

# Clone allows refitting selected model architecture on full data.
from sklearn.base import clone

# Ensemble candidates for robust small-dataset performance.
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)

# Linear baseline model candidate.
from sklearn.linear_model import LogisticRegression

# Evaluation metrics for validation and threshold selection.
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# CV and split utilities for model comparison and calibration.
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split

# Pipeline bundles scaler + estimator into one candidate.
from sklearn.pipeline import Pipeline

# Standardization improves stability for scale-sensitive models.
from sklearn.preprocessing import StandardScaler

# SVM candidate model.
from sklearn.svm import SVC


# Canonical location of the heart dataset relative to this file.
DATASET_PATH = Path(__file__).resolve().parent / "heart-disease.csv"


@dataclass
class ModelBundle:
    """Container for trained model artifacts and evaluation metadata."""

    # Final fitted model object used for inference.
    model: Any

    # Ordered list of training columns expected at inference time.
    feature_columns: List[str]

    # Human-readable name of the selected candidate.
    selected_model_name: str

    # Tuned probability threshold used to convert probas to class labels.
    decision_threshold: float

    # Cross-validation accuracy summary statistics.
    cv_accuracy_mean: float
    cv_accuracy_std: float

    # Cross-validation ROC-AUC summary statistics.
    cv_auc_mean: float
    cv_auc_std: float

    # Held-out validation metrics after threshold tuning.
    validation_accuracy: float
    validation_auc: float

    # Defaults for advanced features in the UI.
    ca_default: int
    thal_default: int

    # Allowed values for advanced dropdowns.
    ca_values: List[int]
    thal_values: List[int]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and validate the expected Cleveland-style heart dataset."""
    # Read CSV into a DataFrame.
    df = pd.read_csv(csv_path)

    # Define the exact schema expected by the pipeline.
    required_columns = {
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    }

    # Find any missing columns.
    missing_columns = required_columns.difference(df.columns)

    # Stop early with a clear schema error if columns are missing.
    if missing_columns:
        missing_cols_display = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_cols_display}")

    # Return validated dataset.
    return df


def build_candidate_models() -> Dict[str, Any]:
    """Create candidate models suitable for small tabular binary classification."""
    # Scaled logistic regression candidate.
    logistic_scaled = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.2,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )

    # Scaled RBF SVC candidate with probability output enabled.
    svc_scaled = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    C=2.0,
                    gamma="scale",
                    kernel="rbf",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    # Random forest candidate.
    random_forest = RandomForestClassifier(
        n_estimators=450,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )

    # Extra trees candidate.
    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )

    # Histogram gradient boosting candidate.
    hist_gb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=250,
        random_state=42,
    )

    # Soft-voting ensemble can smooth variance across individual models.
    soft_vote = VotingClassifier(
        estimators=[
            ("logreg", logistic_scaled),
            ("et", extra_trees),
            ("svc", svc_scaled),
        ],
        voting="soft",
    )

    # Return all candidates keyed by display name.
    return {
        "Soft Voting Ensemble": soft_vote,
        "Extra Trees": extra_trees,
        "Random Forest": random_forest,
        "Logistic Regression (Scaled)": logistic_scaled,
        "SVC RBF (Scaled)": svc_scaled,
        "Hist Gradient Boosting": hist_gb,
    }


def summarize_cv_scores(cv_result: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Convert CV arrays into summary metrics for model comparison."""
    # Package mean/std summaries for both accuracy and AUC.
    return {
        "acc_mean": float(np.mean(cv_result["test_accuracy"])),
        "acc_std": float(np.std(cv_result["test_accuracy"])),
        "auc_mean": float(np.mean(cv_result["test_roc_auc"])),
        "auc_std": float(np.std(cv_result["test_roc_auc"])),
    }


def tune_threshold(y_true: pd.Series, probas: np.ndarray) -> float:
    """Choose a decision threshold using F1-first, accuracy-second ranking."""
    # Scan a practical threshold window instead of fixed 0.50.
    thresholds = np.linspace(0.25, 0.75, 101)

    # Track best candidate threshold and metric values.
    best_threshold = 0.50
    best_f1 = -1.0
    best_acc = -1.0

    # Evaluate each threshold candidate.
    for threshold in thresholds:
        # Convert probabilities to binary predictions at this threshold.
        preds = (probas >= threshold).astype(int)

        # Compute F1 and accuracy for this threshold.
        f1_value = f1_score(y_true, preds, zero_division=0)
        acc_value = accuracy_score(y_true, preds)

        # Prefer higher F1; break ties with higher accuracy.
        is_better = (f1_value > best_f1) or (
            np.isclose(f1_value, best_f1) and (acc_value > best_acc)
        )

        # Update best threshold when improved.
        if is_better:
            best_threshold = float(threshold)
            best_f1 = float(f1_value)
            best_acc = float(acc_value)

    # Return calibrated threshold.
    return best_threshold


def build_optimized_model(csv_path: str) -> ModelBundle:
    """Train and select the best model with repeated CV and threshold tuning."""
    # Load and validate dataset.
    df = load_dataset(csv_path)

    # Split into features and target.
    x = df.drop("target", axis=1)
    y = df["target"]

    # Reserve validation split for threshold tuning and metric reporting.
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Repeated stratified CV is more stable for small datasets.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42)

    # Build the model candidate pool.
    candidate_models = build_candidate_models()

    # Collect summary stats for every candidate.
    model_summaries: List[Tuple[str, Any, Dict[str, float]]] = []

    # Iterate over candidates and score each with repeated CV.
    for model_name, model in candidate_models.items():
        cv_result = cross_validate(
            estimator=model,
            X=x_train,
            y=y_train,
            cv=cv,
            n_jobs=1,
            scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"},
            return_train_score=False,
        )

        # Reduce fold-level arrays to summary metrics.
        summary = summarize_cv_scores(cv_result)

        # Store model name, object, and metrics.
        model_summaries.append((model_name, model, summary))

    # Rank models by AUC first, then by accuracy.
    model_summaries.sort(
        key=lambda row: (row[2]["auc_mean"], row[2]["acc_mean"]),
        reverse=True,
    )

    # Select top-ranked candidate.
    best_name, best_model, best_summary = model_summaries[0]

    # Fit selected model on training split.
    best_model.fit(x_train, y_train)

    # Get validation probabilities for threshold tuning.
    valid_probas = best_model.predict_proba(x_valid)[:, 1]

    # Tune decision threshold on validation data.
    decision_threshold = tune_threshold(y_valid, valid_probas)

    # Apply tuned threshold to produce validation labels.
    valid_preds = (valid_probas >= decision_threshold).astype(int)

    # Compute validation metrics using tuned threshold.
    validation_accuracy = float(accuracy_score(y_valid, valid_preds))
    validation_auc = float(roc_auc_score(y_valid, valid_probas))

    # Refit the selected architecture on all available data.
    final_model = clone(best_model)
    final_model.fit(x, y)

    # Compute defaults for advanced UI fields.
    ca_default = int(df["ca"].mode().iloc[0])
    thal_default = int(df["thal"].mode().iloc[0])

    # Compute allowed discrete values for advanced UI dropdowns.
    ca_values = sorted({int(v) for v in df["ca"].unique().tolist()})
    thal_values = sorted({int(v) for v in df["thal"].unique().tolist()})

    # Return a complete model bundle used by the frontend.
    return ModelBundle(
        model=final_model,
        feature_columns=list(x.columns),
        selected_model_name=best_name,
        decision_threshold=decision_threshold,
        cv_accuracy_mean=best_summary["acc_mean"],
        cv_accuracy_std=best_summary["acc_std"],
        cv_auc_mean=best_summary["auc_mean"],
        cv_auc_std=best_summary["auc_std"],
        validation_accuracy=validation_accuracy,
        validation_auc=validation_auc,
        ca_default=ca_default,
        thal_default=thal_default,
        ca_values=ca_values,
        thal_values=thal_values,
    )


def predict_disease(input_data: Dict[str, float], model_bundle: ModelBundle) -> Tuple[int, float]:
    """Predict label and probability from an encoded feature dictionary."""
    # Reorder user features to exact training-column order.
    ordered_values = [input_data[col] for col in model_bundle.feature_columns]

    # Build one-row DataFrame for scikit-learn inference.
    input_frame = pd.DataFrame([ordered_values], columns=model_bundle.feature_columns)

    # Get positive-class probability.
    probability = float(model_bundle.model.predict_proba(input_frame)[0][1])

    # Convert probability to hard class using tuned threshold.
    label = int(probability >= model_bundle.decision_threshold)

    # Return both hard label and calibrated probability.
    return label, probability
