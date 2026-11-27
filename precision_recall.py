
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay
)

from preprocessing import load_and_prepare_data


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading dataset...")

X, y, preprocessor = load_and_prepare_data()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")


# --------------------------------------------------
# MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}


# --------------------------------------------------
# PRECISION-RECALL COMPARISON
# --------------------------------------------------

plt.figure(figsize=(11, 8))

results = []

for model_name, filename in models.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        print(f"Missing model: {filename}")
        continue

    print(f"\nEvaluating {model_name}...")

    model = joblib.load(model_path)

    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        probabilities
    )

    average_precision = average_precision_score(
        y_test,
        probabilities
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} (AP={average_precision:.3f})"
    )

    results.append({
        "Model": model_name,
        "Average Precision": average_precision
    })


# --------------------------------------------------
# BASELINE
# --------------------------------------------------

baseline = y_test.mean()

plt.axhline(
    y=baseline,
    linestyle="--",
    linewidth=1.5,
    label=f"Baseline ({baseline:.3f})"
)


# --------------------------------------------------
# GRAPH DESIGN
# --------------------------------------------------

plt.title(
    "Precision-Recall Curve Comparison",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)

plt.xlim(0, 1)
plt.ylim(0, 1.05)

plt.grid(alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()


# --------------------------------------------------
# SAVE GRAPH
# --------------------------------------------------

graph_path = OUTPUT_DIR / "precision_recall_comparison.png"

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

if results:

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by="Average Precision",
        ascending=False
    )

    result_df.to_csv(
        OUTPUT_DIR / "precision_recall_results.csv",
        index=False
    )

    print("\nPRECISION-RECALL RESULTS")
    print("=" * 45)

    print(result_df.to_string(index=False))

    print("\nBest Average Precision Model:")
    print(result_df.iloc[0]["Model"])

    print("\nGraph saved:", graph_path)

else:

    print("\nNo models were evaluated.")

print("\nPRECISION-RECALL ANALYSIS COMPLETED!")
