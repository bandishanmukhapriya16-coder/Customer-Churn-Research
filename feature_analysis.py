
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

models = {
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}

for model_name, filename in models.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        print(f"Missing model: {filename}")
        continue

    pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importance = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(15)

    print(f"\n{model_name} - Top 15 Features")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(11, 7))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False
    )

    plt.title(
        f"Top 15 Feature Importances - {model_name}",
        fontsize=15
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Customer Features")
    plt.tight_layout()

    output_name = (
        model_name.lower().replace(" ", "_")
        + "_feature_importance.png"
    )

    plt.savefig(
        OUTPUT_DIR / output_name,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED!")
