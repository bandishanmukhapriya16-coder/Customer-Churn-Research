
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import combinations

from scipy.stats import binomtest

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from preprocessing import load_and_prepare_data


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==================================================
# CONFIGURATION
# ==================================================

RANDOM_STATE = 42
N_BOOTSTRAPS = 1000
CONFIDENCE_LEVEL = 0.95

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}

METRICS = {
    "Accuracy": accuracy_score,
    "Precision": lambda y, p: precision_score(
        y, p, zero_division=0
    ),
    "Recall": lambda y, p: recall_score(
        y, p, zero_division=0
    ),
    "F1 Score": lambda y, p: f1_score(
        y, p, zero_division=0
    )
}


# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading dataset...")

X, y, preprocessor = load_and_prepare_data()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

y_test = np.asarray(y_test)

print("Test samples:", len(y_test))


# ==================================================
# MODEL PREDICTIONS
# ==================================================

predictions = {}
probabilities = {}

print("\nGenerating model predictions...")

for model_name, filename in MODEL_FILES.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        print("Missing model:", filename)
        continue

    model = joblib.load(model_path)

    predictions[model_name] = model.predict(X_test)

    probabilities[model_name] = (
        model.predict_proba(X_test)[:, 1]
    )

    print(model_name, "completed")


if len(predictions) < 2:
    raise RuntimeError(
        "At least two trained models are required."
    )


# ==================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ==================================================

print("\nCalculating bootstrap confidence intervals...")

rng = np.random.default_rng(RANDOM_STATE)

class_zero_indices = np.where(y_test == 0)[0]
class_one_indices = np.where(y_test == 1)[0]

bootstrap_rows = []

for model_name in predictions:

    y_pred = predictions[model_name]
    y_prob = probabilities[model_name]

    for metric_name, metric_function in METRICS.items():

        scores = []

        for _ in range(N_BOOTSTRAPS):

            # Stratified bootstrap sampling
            sampled_zero = rng.choice(
                class_zero_indices,
                size=len(class_zero_indices),
                replace=True
            )

            sampled_one = rng.choice(
                class_one_indices,
                size=len(class_one_indices),
                replace=True
            )

            sampled_indices = np.concatenate(
                [sampled_zero, sampled_one]
            )

            sample_y = y_test[sampled_indices]
            sample_pred = y_pred[sampled_indices]

            score = metric_function(
                sample_y,
                sample_pred
            )

            scores.append(score)

        lower = np.percentile(
            scores,
            2.5
        )

        upper = np.percentile(
            scores,
            97.5
        )

        original_score = metric_function(
            y_test,
            y_pred
        )

        bootstrap_rows.append({
            "Model": model_name,
            "Metric": metric_name,
            "Score": original_score,
            "Lower 95% CI": lower,
            "Upper 95% CI": upper
        })

    # ROC-AUC and Average Precision
    for metric_name, metric_function in [
        ("ROC-AUC", roc_auc_score),
        ("Average Precision", average_precision_score)
    ]:

        scores = []

        for _ in range(N_BOOTSTRAPS):

            sampled_zero = rng.choice(
                class_zero_indices,
                size=len(class_zero_indices),
                replace=True
            )

            sampled_one = rng.choice(
                class_one_indices,
                size=len(class_one_indices),
                replace=True
            )

            sampled_indices = np.concatenate(
                [sampled_zero, sampled_one]
            )

            sample_y = y_test[sampled_indices]
            sample_prob = y_prob[sampled_indices]

            score = metric_function(
                sample_y,
                sample_prob
            )

            scores.append(score)

        lower = np.percentile(scores, 2.5)
        upper = np.percentile(scores, 97.5)

        original_score = metric_function(
            y_test,
            y_prob
        )

        bootstrap_rows.append({
            "Model": model_name,
            "Metric": metric_name,
            "Score": original_score,
            "Lower 95% CI": lower,
            "Upper 95% CI": upper
        })


bootstrap_df = pd.DataFrame(bootstrap_rows)

bootstrap_path = (
    OUTPUT_DIR / "bootstrap_confidence_intervals.csv"
)

bootstrap_df.to_csv(
    bootstrap_path,
    index=False
)


# ==================================================
# MCMEMAR'S EXACT TEST
# ==================================================

print("\nRunning pairwise McNemar tests...")

pairwise_rows = []

model_names = list(predictions.keys())

total_comparisons = len(
    list(combinations(model_names, 2))
)

for model_a, model_b in combinations(model_names, 2):

    pred_a = predictions[model_a]
    pred_b = predictions[model_b]

    correct_a = pred_a == y_test
    correct_b = pred_b == y_test

    # Model A correct, Model B incorrect
    b = int(np.sum(correct_a & ~correct_b))

    # Model A incorrect, Model B correct
    c = int(np.sum(~correct_a & correct_b))

    discordant = b + c

    if discordant == 0:

        p_value = 1.0

    else:

        p_value = binomtest(
            k=min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    adjusted_p = min(
        p_value * total_comparisons,
        1.0
    )

    pairwise_rows.append({
        "Model A": model_a,
        "Model B": model_b,
        "A Correct B Wrong": b,
        "A Wrong B Correct": c,
        "Discordant Predictions": discordant,
        "Raw P Value": p_value,
        "Bonferroni Adjusted P Value": adjusted_p,
        "Significant at 0.05": adjusted_p < 0.05
    })


pairwise_df = pd.DataFrame(pairwise_rows)

pairwise_path = OUTPUT_DIR / "mcnemar_test_results.csv"

pairwise_df.to_csv(
    pairwise_path,
    index=False
)


# ==================================================
# PRINT RESULTS
# ==================================================

print("\n" + "=" * 70)
print("BOOTSTRAP CONFIDENCE INTERVAL RESULTS")
print("=" * 70)

print(
    bootstrap_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print("PAIRWISE MCMEMAR TEST RESULTS")
print("=" * 70)

print(
    pairwise_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# ==================================================
# SAVE TEXT REPORT
# ==================================================

report_path = OUTPUT_DIR / "statistical_analysis_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("CUSTOMER CHURN STATISTICAL ANALYSIS\n")
    file.write("=" * 60)

    file.write("\n\n1. BOOTSTRAP CONFIDENCE INTERVALS\n")
    file.write("-" * 60)

    file.write(
        bootstrap_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    file.write("\n\n2. PAIRWISE MCMEMAR TESTS\n")
    file.write("-" * 60)

    file.write(
        pairwise_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    file.write("\n\n3. INTERPRETATION\n")
    file.write("-" * 60)

    file.write(
        "\nBootstrap intervals estimate uncertainty "
        "in the observed test-set metrics."
    )

    file.write(
        "\nMcNemar's test compares paired classification "
        "errors between two models."
    )

    file.write(
        "\nBonferroni-adjusted p-values account for "
        "the multiple pairwise comparisons."
    )

    file.write(
        "\nA p-value below 0.05 after correction indicates "
        "evidence of different error patterns under this test."
    )

    file.write(
        "\nStatistical significance does not necessarily "
        "mean the performance difference is practically important."
    )

    file.write(
        "\nThese findings are conditional on this particular "
        "train-test split and should be validated on independent data."
    )

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS COMPLETED!")
print("=" * 60)

print("\nFiles saved:")
print(bootstrap_path)
print(pairwise_path)
print(report_path)
