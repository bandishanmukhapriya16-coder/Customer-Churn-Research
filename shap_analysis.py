
import matplotlib
matplotlib.use("Agg")

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split


# Paths
OUTPUT_DIR = Path("outputs")
MODEL_PATH = Path("models/random_forest.pkl")

OUTPUT_DIR.mkdir(exist_ok=True)


def main():

    print("\nStarting SHAP explainability analysis...")

    # Load dataset
    df = pd.read_csv(
        "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.drop(columns=["customerID"])

    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["Churn"])

    # Same test split used during model training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Load trained Random Forest pipeline
    model = joblib.load(MODEL_PATH)

    # Identify preprocessing and classifier steps
    preprocessor = next(
        step for step in model.named_steps.values()
        if hasattr(step, "get_feature_names_out")
    )

    classifier = list(model.named_steps.values())[-1]

    # Transform test data
    X_transformed = preprocessor.transform(X_test)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    # Explain a manageable sample
    sample_size = min(150, len(X_test))

    X_sample = X_transformed[:sample_size]

    print("Calculating SHAP values...")
    print("Samples:", sample_size)
    print("Features:", len(feature_names))

    # Tree SHAP explainer
    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer(X_sample)

    # Select positive class: Churn = Yes
    if len(shap_values.values.shape) == 3:
        positive_values = shap_values.values[:, :, 1]
        positive_base = shap_values.base_values[:, 1]
    else:
        positive_values = shap_values.values
        positive_base = shap_values.base_values

    # Create a SHAP Explanation object
    explanation = shap.Explanation(
        values=positive_values,
        base_values=positive_base,
        data=X_sample,
        feature_names=list(feature_names)
    )

    # 1. SHAP Summary Plot
    print("Creating SHAP summary plot...")

    plt.figure(figsize=(12, 8))

    shap.plots.beeswarm(
        explanation,
        max_display=15,
        show=False
    )

    plt.title(
        "SHAP Summary - Customer Churn Prediction"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # 2. SHAP Feature Importance Plot
    print("Creating feature importance plot...")

    plt.figure(figsize=(12, 8))

    shap.plots.bar(
        explanation,
        max_display=15,
        show=False
    )

    plt.title(
        "SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # 3. Individual Customer Waterfall Plot
    print("Creating individual prediction explanation...")

    plt.figure(figsize=(12, 8))

    shap.plots.waterfall(
        explanation[0],
        max_display=15,
        show=False
    )

    plt.title(
        "SHAP Waterfall - Individual Customer"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Save numerical feature importance
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": abs(positive_values).mean(axis=0)
    })

    importance = importance.sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False
    )

    print("\nSHAP ANALYSIS COMPLETED!")
    print("\nGenerated files:")

    print("shap_summary.png")
    print("shap_feature_importance.png")
    print("shap_waterfall.png")
    print("shap_feature_importance.csv")


if __name__ == "__main__":
    main()
