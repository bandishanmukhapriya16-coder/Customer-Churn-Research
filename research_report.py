
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULT_PATH = BASE_DIR / "outputs" / "model_comparison.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "research_results_report.txt"


def generate_report():

    print("\nCUSTOMER CHURN RESEARCH REPORT")
    print("=" * 55)

    # Check required files
    if not DATA_PATH.exists():
        print("Dataset not found!")
        return

    if not RESULT_PATH.exists():
        print("Model comparison file not found!")
        print("Run python train.py first.")
        return

    # Load dataset and results
    df = pd.read_csv(DATA_PATH)
    results = pd.read_csv(RESULT_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Dataset statistics
    total_customers = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = (df["Churn"] == "No").sum()
    churn_percentage = churned / total_customers * 100

    # Identify model column
    model_column = next(
        (
            col for col in results.columns
            if "model" in col.lower() or "algorithm" in col.lower()
        ),
        results.columns[0]
    )

    # Identify metric columns
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    available_metrics = [
        col for col in results.columns
        if col.lower().replace(" ", "_") in metrics
    ]

    if not available_metrics:
        print("No evaluation metrics found in comparison CSV.")
        print("Check outputs/model_comparison.csv")
        return

    # Report content
    report = []

    report.append("CUSTOMER CHURN PREDICTION")
    report.append("MACHINE LEARNING RESEARCH REPORT")
    report.append("=" * 60)

    report.append(
        "\nGenerated on: " +
        datetime.now().strftime("%d-%m-%Y %H:%M")
    )

    report.append("\n1. DATASET INFORMATION")
    report.append("-" * 40)
    report.append(f"Dataset: IBM Telco Customer Churn")
    report.append(f"Total Customers: {total_customers}")
    report.append(f"Churned Customers: {churned}")
    report.append(f"Retained Customers: {retained}")
    report.append(f"Churn Percentage: {churn_percentage:.2f}%")
    report.append(f"Total Features: {df.shape[1] - 1}")

    report.append("\n2. EXPERIMENTAL MODELS")
    report.append("-" * 40)

    for model in results[model_column]:
        report.append(f"- {model}")

    report.append("\n3. MODEL PERFORMANCE")
    report.append("-" * 40)

    report.append(
        results.to_string(index=False, float_format="%.4f")
    )

    report.append("\n4. BEST MODEL FOR EACH METRIC")
    report.append("-" * 40)

    for metric in available_metrics:

        values = pd.to_numeric(
            results[metric], errors="coerce"
        )

        valid = values.dropna()

        if valid.empty:
            continue

        best_index = valid.idxmax()

        best_model = results.loc[best_index, model_column]
        best_score = values.loc[best_index]

        report.append(
            f"{metric}: {best_model} ({best_score:.4f})"
        )

    report.append("\n5. RESEARCH OBSERVATIONS")
    report.append("-" * 40)

    report.append(
        "The experiment compares multiple classification "
        "algorithms on the same customer churn dataset."
    )

    report.append(
        "Accuracy, precision, recall, F1-score and ROC-AUC "
        "are used to evaluate predictive performance."
    )

    report.append(
        "Different evaluation metrics measure different "
        "aspects of model performance."
    )

    report.append(
        "The best algorithm may differ depending on the "
        "evaluation metric being considered."
    )

    report.append("\n6. CONCLUSION")
    report.append("-" * 40)

    report.append(
        "This experiment demonstrates how machine learning "
        "can be applied to customer churn prediction."
    )

    report.append(
        "The final model selection should consider multiple "
        "evaluation metrics rather than accuracy alone."
    )

    report.append(
        "The findings are limited to this dataset and "
        "experimental setup."
    )

    report.append("\n" + "=" * 60)
    report.append("END OF RESEARCH REPORT")

    # Save report
    final_report = "\n".join(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(final_report)

    print(final_report)

    print("\nReport saved successfully!")
    print("Location:", REPORT_PATH)


if __name__ == "__main__":
    generate_report()
