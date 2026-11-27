
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
# Project paths
DATA_PATH = Path("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Theme
sns.set_theme(style="darkgrid", palette="husl")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\n========== DATASET OVERVIEW ==========")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

print("\n========== DATA INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all").T)

print("\n========== CHURN DISTRIBUTION ==========")
print(df["Churn"].value_counts())

# 1. Churn distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Customer Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_churn_distribution.png")
plt.show()

# 2. Churn by contract
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_contract_churn.png")
plt.show()

# 3. Tenure distribution
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df, x="tenure", hue="Churn",
    bins=25, multiple="stack"
)
plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_tenure_distribution.png")
plt.show()

# 4. Monthly charges
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_monthly_charges.png")
plt.show()

# 5. Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Numerical Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_correlation_heatmap.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Graphs saved inside outputs folder.")
