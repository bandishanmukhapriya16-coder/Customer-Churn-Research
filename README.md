# 📊 Customer Churn Prediction Using Machine Learning

An end-to-end machine learning research project that analyzes customer churn patterns, compares classification algorithms, evaluates predictive performance, and explains model behaviour through Explainable AI.

## 🌟 Project Overview

Customer churn occurs when customers discontinue their services. Predicting churn can help organizations understand customer behaviour and identify customers who may be at risk of leaving.

This project uses the IBM Telco Customer Churn dataset to compare four machine learning algorithms and investigate their performance through statistical evaluation, cross-validation, probability calibration, and error analysis.

## 🎯 Objectives

- Analyze customer churn patterns.
- Perform exploratory data analysis.
- Build and compare four machine learning models.
- Evaluate models using multiple performance metrics.
- Apply five-fold stratified cross-validation.
- Investigate model calibration.
- Explain model predictions using SHAP.
- Analyze false positives and false negatives.
- Build an interactive Streamlit dashboard.

## 📁 Dataset

**Dataset:** IBM Telco Customer Churn

- Total records: 7,043
- Churned customers: 1,869
- Retained customers: 5,174
- Features: Customer demographics, services, contract details, and billing information.
- Target: `Churn`

## 🤖 Machine Learning Algorithms

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 84.19% |
| Decision Tree | 72.89% | 48.96% | 50.53% | 49.74% | 65.73% |
| Random Forest | 77.57% | 60.00% | 46.52% | 52.41% | 81.86% |
| Gradient Boosting | 80.62% | 67.35% | 52.41% | 58.95% | 84.34% |

## 🔬 Research Experiments

| Experiment | Description |
|---|---|
| Exploratory Data Analysis | Customer churn patterns and distributions |
| Model Comparison | Evaluation of four classification algorithms |
| Statistical Analysis | Bootstrap confidence intervals and McNemar tests |
| Explainable AI | SHAP-based model interpretation |
| Cross-Validation | Five-fold stratified evaluation |
| Model Calibration | Brier Score and Log Loss |
| Error Analysis | False positives, false negatives, and subgroup error patterns |

## 🖥️ Interactive Dashboard

The Streamlit dashboard contains ten sections:

1. Overview
2. Exploratory Analysis
3. Model Comparison
4. Research Insights
5. Statistical Analysis
6. Explainable AI
7. Cross-Validation
8. Model Calibration
9. Customer Prediction
10. Comparative Error Analysis

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- SHAP
- SciPy
- Joblib

## 🚀 Installation and Execution

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd Customer_Churn_Research
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the machine learning models:

```bash
python train.py
```

Run the research dashboard:

```bash
streamlit run app.py
```

## 📂 Project Structure

```text
Customer_Churn_Research/
│
├── dataset/
├── models/
├── outputs/
│
├── app.py
├── train.py
├── preprocessing.py
├── eda.py
├── evaluate.py
├── feature_analysis.py
├── research_report.py
├── precision_recall.py
├── statistical_analysis.py
├── shap_analysis.py
├── cross_validation.py
├── model_calibration.py
├── error_analysis.py
│
├── requirements.txt
├── FINAL_RESEARCH_REPORT.md
└── README.md
```

## 📊 Key Findings

- Gradient Boosting achieved the highest held-out accuracy of 80.62%.
- Gradient Boosting achieved the highest held-out ROC-AUC of 84.34%.
- Logistic Regression achieved the highest held-out recall of 55.88%.
- Logistic Regression achieved the highest held-out F1-score of 60.40%.
- Gradient Boosting achieved the lowest Brier Score and Log Loss.

## ⚠️ Limitations

The results are specific to the selected dataset and experimental setup. Further validation on independent datasets is necessary before applying these models in real-world customer retention systems.

## 🔮 Future Improvements

- Hyperparameter optimization
- Additional ensemble models
- Classification threshold optimization
- External dataset validation
- Advanced fairness evaluation
- Model deployment and monitoring

## 👩‍💻 Project Information

**Project:** Customer Churn Prediction Using Machine Learning  
**Domain:** Artificial Intelligence and Data Science  
**Type:** Research-oriented Machine Learning Project

---

*This project is developed for educational and research purposes.*