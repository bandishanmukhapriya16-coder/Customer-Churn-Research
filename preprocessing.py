
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


DATA_PATH = Path("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_and_prepare_data():

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    df = df.drop(columns=["customerID"])

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    X = df.drop(columns=["Churn"])

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
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

    print("\nPreprocessing completed!")
    print("Rows:", len(X))
    print("Features:", X.shape[1])
    print("Missing values:", X.isnull().sum().sum())

    return X, y, preprocessor


if __name__ == "__main__":
    X, y, preprocessor = load_and_prepare_data()
