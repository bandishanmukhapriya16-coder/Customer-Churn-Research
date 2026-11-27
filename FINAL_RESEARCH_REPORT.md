
# Customer Churn Prediction Using Machine Learning

## Abstract

This research investigates customer churn prediction using four supervised machine learning algorithms: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. The IBM Telco Customer Churn dataset containing 7,043 customer records was used. The models were evaluated using accuracy, precision, recall, F1-score, ROC-AUC, five-fold stratified cross-validation, probability calibration, and comparative error analysis.

The results show that Gradient Boosting achieved the highest held-out accuracy of 80.62% and ROC-AUC of 84.34%. Logistic Regression achieved the highest held-out recall of 55.88% and F1-score of 60.40%. Cross-validation further demonstrated that the two algorithms offer competitive but different strengths.

## 1. Introduction

Customer churn occurs when customers discontinue their relationship with a service provider. Predicting churn can help organizations identify customers who may be at risk of leaving.

Machine learning provides methods for analyzing customer information and identifying patterns associated with churn.

This research compares four classification algorithms to understand their predictive performance and reliability.

## 2. Dataset Description

Dataset: IBM Telco Customer Churn

- Total customer records: 7,043
- Churned customers: 1,869
- Retained customers: 5,174
- Total original columns: 21
- Target variable: Churn
- Churn percentage: 26.54%

The customer information includes tenure, contract type, monthly charges, internet service, payment method, and other service-related attributes.

## 3. Data Preprocessing

The preprocessing workflow includes:

- Converting TotalCharges into numeric format.
- Handling missing numerical values using median imputation.
- Handling missing categorical values using most-frequent imputation.
- Scaling numerical features.
- Applying one-hot encoding to categorical features.
- Removing the customerID identifier from model features.
- Encoding the target variable as binary values.

The dataset was divided into training and testing subsets using an 80:20 stratified split for the main model comparison.

## 4. Machine Learning Algorithms

Four supervised classification algorithms were evaluated.

### 4.1 Logistic Regression

A linear classification algorithm used as a baseline model.

### 4.2 Decision Tree

A tree-based algorithm that learns decision rules from input features.

### 4.3 Random Forest

An ensemble algorithm that combines predictions from multiple decision trees.

### 4.4 Gradient Boosting

An ensemble algorithm that builds sequential decision trees to improve predictive performance.

## 5. Experimental Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Brier Score
- Log Loss
- Confusion matrix analysis

### 5.1 Held-Out Test Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 84.19% |
| Decision Tree | 72.89% | 48.96% | 50.53% | 49.74% | 65.73% |
| Random Forest | 77.57% | 60.00% | 46.52% | 52.41% | 81.86% |
| Gradient Boosting | 80.62% | 67.35% | 52.41% | 58.95% | 84.34% |

Gradient Boosting achieved the highest accuracy, precision, and ROC-AUC. Logistic Regression achieved the highest recall and F1-score.

### 5.2 Five-Fold Cross-Validation

| Model | Mean Accuracy | Mean F1-score | Mean ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 80.52% | 60.08% | 84.50% |
| Decision Tree | 73.01% | 49.86% | 66.00% |
| Random Forest | 78.53% | 54.38% | 82.12% |
| Gradient Boosting | 80.49% | 58.82% | 84.71% |

Cross-validation results support the competitive performance of Logistic Regression and Gradient Boosting.

### 5.3 Probability Calibration

| Model | Brier Score | Log Loss |
|---|---:|---:|
| Logistic Regression | 0.1381 | 0.4205 |
| Decision Tree | 0.2713 | 9.3646 |
| Random Forest | 0.1472 | 0.5164 |
| Gradient Boosting | 0.1359 | 0.4178 |

Gradient Boosting achieved the lowest Brier Score and Log Loss among the tested models.

### 5.4 Comparative Error Analysis

| Model | Total Errors | Error Rate |
|---|---:|---:|
| Logistic Regression | 274 | 19.45% |
| Decision Tree | 382 | 27.11% |
| Random Forest | 316 | 22.43% |
| Gradient Boosting | 273 | 19.38% |

Gradient Boosting produced the fewest errors, with Logistic Regression very close behind.

## 6. Research Discussion

The results demonstrate that model performance depends on the evaluation metric.

Gradient Boosting achieved the strongest held-out accuracy and probability calibration results. Its ROC-AUC was also the highest in both the held-out evaluation and cross-validation.

Logistic Regression achieved a stronger recall and F1-score. This indicates that it identified a greater proportion of actual churn cases at the selected classification threshold.

The Decision Tree showed weaker performance across most metrics. Random Forest achieved competitive ROC-AUC but had comparatively lower recall.

These findings demonstrate the importance of evaluating classification models using multiple performance measures rather than accuracy alone.

## 7. Explainable AI

SHAP analysis was included to investigate how input features contribute to model predictions.

Feature importance analysis was also conducted for tree-based models.

These techniques provide insight into model behaviour. However, feature importance and SHAP values do not establish causal relationships.

## 8. Limitations

- The research uses a single publicly available telecom dataset.
- Results may not generalize to other industries or customer populations.
- Model performance depends on the selected preprocessing and classification threshold.
- Cross-validation and held-out evaluation provide estimates rather than guarantees of future performance.
- Group-level error differences require further investigation before drawing fairness conclusions.
- Additional external validation would strengthen the findings.

## 9. Conclusion

This research compared four machine learning algorithms for predicting customer churn using the IBM Telco Customer Churn dataset.

Gradient Boosting achieved the highest held-out accuracy of 80.62%, while Logistic Regression achieved the highest held-out F1-score of 60.40%.

Five-fold cross-validation confirmed that both algorithms are competitive. Gradient Boosting achieved the highest mean ROC-AUC, while Logistic Regression achieved the highest mean F1-score.

The research demonstrates that machine learning can support customer churn analysis and that reliable model selection requires multiple evaluation metrics, validation techniques, and error analysis.

## 10. Future Work

Future research could investigate:

- Hyperparameter optimization.
- Alternative classification thresholds.
- Class imbalance handling.
- Additional ensemble algorithms.
- External dataset validation.
- More detailed fairness and subgroup evaluation.
- Deployment monitoring and model drift.

---

**Project:** Customer Churn Prediction Using Machine Learning

**Dataset:** IBM Telco Customer Churn

**Algorithms:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
