
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

from preprocessing import load_and_prepare_data


# Create output folder
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)


# Models trained in train.py
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}


def main():

    print("\nStarting model evaluation...")

    # Load dataset
    X, y, preprocessor = load_and_prepare_data()

    # Use the same split as train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Store ROC values
    plt.figure(figsize=(10, 7))

    sns.set_theme(style="whitegrid")

    colors = [
        "#8B5CF6",
        "#06B6D4",
        "#10B981",
        "#F59E0B"
    ]

    models_loaded = 0

    for (model_name, filename), color in zip(
        MODEL_FILES.items(), colors
    ):

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        print(f"\nEvaluating {model_name}...")

        # Load saved pipeline
        model = joblib.load(model_path)

        # Predictions
        y_pred = model.predict(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(7, 5))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Churn", "Churn"]
        )

        display.plot(
            ax=ax,
            cmap="Purples",
            colorbar=False
        )

        ax.set_title(
            f"{model_name} - Confusion Matrix",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")

        confusion_path = (
            OUTPUT_DIR / f"{safe_name}_confusion_matrix.png"
        )

        fig.savefig(
            confusion_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Saved: {confusion_path}")

        # Get probability scores for ROC curve
        if hasattr(model, "predict_proba"):

            y_scores = model.predict_proba(X_test)[:, 1]

        elif hasattr(model, "decision_function"):

            y_scores = model.decision_function(X_test)

        else:
            print(f"Cannot calculate ROC for {model_name}")
            continue

        # ROC curve
        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_scores
        )

        auc_score = roc_auc_score(
            y_test,
            y_scores
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2.5,
            color=color,
            label=f"{model_name} (AUC = {auc_score:.3f})"
        )

        models_loaded += 1

        print(f"AUC: {auc_score:.4f}")

    # Add ROC reference line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Random Classifier"
    )

    plt.title(
        "ROC Curve Comparison - Customer Churn Prediction",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "False Positive Rate",
        fontsize=12
    )

    plt.ylabel(
        "True Positive Rate",
        fontsize=12
    )

    plt.legend(
        loc="lower right",
        fontsize=10
    )

    plt.grid(alpha=0.25)

    plt.tight_layout()

    # Save ROC image
    roc_path = OUTPUT_DIR / "all_models_roc_comparison.png"

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETED")
    print("=" * 50)

    print(f"Models evaluated: {models_loaded}")
    print(f"ROC curve saved at: {roc_path}")

    if models_loaded == 0:
        print("No models were evaluated. Check your models folder.")


if __name__ == "__main__":
    main()
