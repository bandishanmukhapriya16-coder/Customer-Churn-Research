
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

customer_ids = df["customerID"].copy()

df = df.drop(columns=["customerID"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=["Churn"])


# =====================================================
# TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    customer_ids,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================================
# MODEL LIST
# =====================================================

models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}


# =====================================================
# RESULT STORAGE
# =====================================================

summary_results = []
detailed_results = []

group_results = []

error_categories = {}


# =====================================================
# MODEL ERROR ANALYSIS
# =====================================================

for model_name, filename in models.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():

        print(f"Model not found: {filename}")
        continue

    print(f"\nAnalysing {model_name}...")

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    total = len(y_test)

    error_rate = (fp + fn) / total

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    summary_results.append({
        "Model": model_name,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
        "Total Errors": fp + fn,
        "Error Rate": error_rate,
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate
    })

    error_categories[model_name] = {
        "False Positive": fp,
        "False Negative": fn,
        "True Positive": tp,
        "True Negative": tn
    }

    # Detailed customer-level predictions
    model_details = pd.DataFrame({
        "CustomerID": id_test.values,
        "Actual": y_test.values,
        "Predicted": predictions
    })

    model_details["Model"] = model_name

    model_details["Actual Label"] = (
        model_details["Actual"]
        .map({0: "No Churn", 1: "Churn"})
    )

    model_details["Predicted Label"] = (
        model_details["Predicted"]
        .map({0: "No Churn", 1: "Churn"})
    )

    model_details["Prediction Result"] = np.select(
        [
            (model_details["Actual"] == 1) &
            (model_details["Predicted"] == 1),

            (model_details["Actual"] == 0) &
            (model_details["Predicted"] == 0),

            (model_details["Actual"] == 0) &
            (model_details["Predicted"] == 1),

            (model_details["Actual"] == 1) &
            (model_details["Predicted"] == 0)
        ],
        [
            "True Positive",
            "True Negative",
            "False Positive",
            "False Negative"
        ],
        default="Unknown"
    )

    detailed_results.append(model_details)

    # Group analysis
    group_test = X_test.copy()

    group_test["Actual"] = y_test.values
    group_test["Predicted"] = predictions

    group_test["Error"] = (
        group_test["Actual"] != group_test["Predicted"]
    ).astype(int)

    for feature in [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "SeniorCitizen",
        "PaperlessBilling",
        "Partner"
    ]:

        if feature not in group_test.columns:
            continue

        grouped = (
            group_test.groupby(feature, dropna=False)
            .agg(
                Customers=("Error", "size"),
                Errors=("Error", "sum")
            )
            .reset_index()
        )

        grouped["Error Rate"] = (
            grouped["Errors"] / grouped["Customers"]
        )

        grouped["Feature"] = feature
        grouped["Model"] = model_name

        grouped = grouped.rename(
            columns={feature: "Group"}
        )

        group_results.append(grouped)


# =====================================================
# SAVE CSV FILES
# =====================================================

if not summary_results:

    raise RuntimeError(
        "No trained models found. Run train.py first."
    )

summary_df = pd.DataFrame(summary_results)

summary_df.to_csv(
    OUTPUT_DIR / "error_analysis_summary.csv",
    index=False
)

details_df = pd.concat(
    detailed_results,
    ignore_index=True
)

details_df.to_csv(
    OUTPUT_DIR / "customer_error_details.csv",
    index=False
)

group_df = pd.concat(
    group_results,
    ignore_index=True
)

group_df.to_csv(
    OUTPUT_DIR / "group_error_analysis.csv",
    index=False
)


# =====================================================
# VISUALIZATION 1: ERROR COUNTS
# =====================================================

error_plot = summary_df[
    ["Model", "False Positives", "False Negatives"]
].melt(
    id_vars="Model",
    var_name="Error Type",
    value_name="Count"
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=error_plot,
    x="Model",
    y="Count",
    hue="Error Type"
)

plt.title("False Positive vs False Negative Comparison")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_type_comparison.png",
    dpi=300
)

plt.close()


# =====================================================
# VISUALIZATION 2: ERROR RATE
# =====================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=summary_df,
    x="Model",
    y="Error Rate"
)

plt.title("Overall Prediction Error Rate")
plt.ylabel("Error Rate")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "overall_error_rate.png",
    dpi=300
)

plt.close()


# =====================================================
# VISUALIZATION 3: CONFUSION MATRIX COMPARISON
# =====================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes = axes.flatten()

for index, row in summary_df.iterrows():

    matrix = np.array([
        [
            row["True Negatives"],
            row["False Positives"]
        ],
        [
            row["False Negatives"],
            row["True Positives"]
        ]
    ])

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        cbar=False,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=axes[index]
    )

    axes[index].set_title(row["Model"])
    axes[index].set_xlabel("Predicted")
    axes[index].set_ylabel("Actual")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_confusion_comparison.png",
    dpi=300
)

plt.close()


# =====================================================
# VISUALIZATION 4: CUSTOMER GROUP ERROR RATES
# =====================================================

group_plot = group_df[
    group_df["Feature"].isin([
        "Contract",
        "InternetService",
        "PaymentMethod"
    ])
].copy()

if not group_plot.empty:

    group_plot["Category"] = (
        group_plot["Feature"].astype(str)
        + ": "
        + group_plot["Group"].astype(str)
    )

    plt.figure(figsize=(14, 8))

    sns.barplot(
        data=group_plot,
        x="Category",
        y="Error Rate",
        hue="Model"
    )

    plt.title("Prediction Error Rates Across Customer Groups")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "customer_group_error_rates.png",
        dpi=300
    )

    plt.close()


# =====================================================
# FINAL REPORT
# =====================================================

with open(
    OUTPUT_DIR / "error_analysis_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("CUSTOMER CHURN ERROR ANALYSIS\n")
    file.write("=" * 55 + "\n\n")

    file.write("MODEL ERROR SUMMARY\n\n")

    file.write(
        summary_df.to_string(index=False)
    )

    file.write("\n\nINTERPRETATION\n")

    best_error_model = summary_df.loc[
        summary_df["Error Rate"].idxmin()
    ]

    file.write(
        f"\nLowest overall error rate: "
        f"{best_error_model['Model']}\n"
    )

    file.write(
        f"Error rate: "
        f"{best_error_model['Error Rate']:.4f}\n"
    )

    file.write(
        "\nFalse positives are customers predicted to churn "
        "who actually did not churn.\n"
    )

    file.write(
        "False negatives are customers predicted not to churn "
        "who actually churned.\n"
    )

    file.write(
        "\nGroup error rates are descriptive and do not establish "
        "that customer characteristics cause prediction errors.\n"
    )


# =====================================================
# COMPLETION
# =====================================================

print("\nERROR ANALYSIS COMPLETED!")

print("\nModel Error Summary:")
print(summary_df.to_string(index=False))

print("\nGenerated files:")

for filename in [
    "error_analysis_summary.csv",
    "customer_error_details.csv",
    "group_error_analysis.csv",
    "error_type_comparison.png",
    "overall_error_rate.png",
    "error_confusion_comparison.png",
    "customer_group_error_rates.png",
    "error_analysis_report.txt"
]:

    if (OUTPUT_DIR / filename).exists():
        print("✓", filename)

