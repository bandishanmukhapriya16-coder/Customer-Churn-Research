
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# -------------------------------------------------
# PATHS
# -------------------------------------------------

DATA_PATH = Path(
    "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

print("\nLoading dataset...")

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

print("Dataset shape:", X.shape)


# -------------------------------------------------
# PREPROCESSING
# -------------------------------------------------

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# -------------------------------------------------
# DEFINE MODELS
# -------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# -------------------------------------------------
# CROSS VALIDATION
# -------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = []

fold_results = []

print("\nStarting 5-Fold Cross-Validation...")

for model_name, classifier in models.items():

    print(f"\nEvaluating: {model_name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    row = {
        "Model": model_name
    }

    for metric in scoring:

        values = scores[f"test_{metric}"]

        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()

        for fold_number, value in enumerate(values, start=1):

            fold_results.append({
                "Model": model_name,
                "Fold": fold_number,
                "Metric": metric,
                "Score": value
            })

    row["fit_time_mean"] = scores["fit_time"].mean()

    results.append(row)

    print("Accuracy:", round(row["accuracy_mean"], 4))
    print("Precision:", round(row["precision_mean"], 4))
    print("Recall:", round(row["recall_mean"], 4))
    print("F1:", round(row["f1_mean"], 4))
    print("ROC-AUC:", round(row["roc_auc_mean"], 4))


# -------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------

results_df = pd.DataFrame(results)

fold_df = pd.DataFrame(fold_results)

results_df.to_csv(
    OUTPUT_DIR / "cross_validation_results.csv",
    index=False
)

fold_df.to_csv(
    OUTPUT_DIR / "cross_validation_fold_results.csv",
    index=False
)


# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------

sns.set_theme(style="whitegrid")

plot_df = results_df.melt(
    id_vars=["Model"],
    value_vars=[
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean"
    ],
    var_name="Metric",
    value_name="Score"
)

plot_df["Metric"] = (
    plot_df["Metric"]
    .str.replace("_mean", "", regex=False)
    .str.upper()
)

plt.figure(figsize=(13, 7))

sns.barplot(
    data=plot_df,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title(
    "5-Fold Cross-Validation Model Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Mean Score")
plt.ylim(0, 1)

plt.xticks(rotation=15)
plt.legend(title="Evaluation Metric")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cross_validation_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------------------
# FOLD VARIABILITY VISUALIZATION
# -------------------------------------------------

plt.figure(figsize=(12, 7))

sns.boxplot(
    data=fold_df[fold_df["Metric"] == "f1"],
    x="Model",
    y="Score"
)

plt.title(
    "F1-Score Distribution Across Five Folds",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Machine Learning Model")
plt.ylabel("F1-Score")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cross_validation_f1_stability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------------------
# FINAL OUTPUT
# -------------------------------------------------

print("\n" + "=" * 55)
print("CROSS-VALIDATION COMPLETED")
print("=" * 55)

print("\nMean Scores and Standard Deviations:")

print(
    results_df[
        [
            "Model",
            "accuracy_mean",
            "accuracy_std",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std"
        ]
    ].to_string(index=False)
)

print("\nGenerated files:")
print("cross_validation_results.csv")
print("cross_validation_fold_results.csv")
print("cross_validation_comparison.png")
print("cross_validation_f1_stability.png")
