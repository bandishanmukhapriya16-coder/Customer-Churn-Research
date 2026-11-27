
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

import joblib


# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# Load dataset
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.drop(columns=["customerID"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=["Churn"])


# Recreate the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Models
models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}


# Store results
results = []


# Calibration plot
plt.figure(figsize=(10, 8))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Perfect Calibration"
)


# Evaluate models
for model_name, filename in models.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():

        print(f"Model not found: {filename}")
        continue

    print(f"Evaluating {model_name}...")

    model = joblib.load(model_path)

    probabilities = model.predict_proba(X_test)[:, 1]

    probabilities = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15
    )

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    loss = log_loss(
        y_test,
        probabilities
    )

    fraction_positive, mean_probability = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="uniform"
    )

    plt.plot(
        mean_probability,
        fraction_positive,
        marker="o",
        linewidth=2,
        label=model_name
    )

    results.append({
        "Model": model_name,
        "Brier Score": brier,
        "Log Loss": loss
    })


# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "calibration_metrics.csv",
    index=False
)


# Save calibration curve
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positive Cases")

plt.title(
    "Model Calibration Comparison - Customer Churn"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "calibration_comparison.png",
    dpi=300
)

plt.close()


print("\nCalibration analysis completed!")

print(results_df)

print("\nGenerated files:")
print("outputs/calibration_metrics.csv")
print("outputs/calibration_comparison.png")
