
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080d1b, #101a35, #17102b);
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111a33, #21143d);
    border-right: 1px solid #35446b;
}

h1, h2, h3 {
    color: #a5b4fc !important;
}

p, label, span {
    color: #e2e8f0;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e2b50, #30204f);
    border: 1px solid #53618d;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricValue"] {
    color: #67e8f9;
}

div[data-testid="stMetricLabel"] {
    color: #c4b5fd;
}

.stButton button {
    background: linear-gradient(90deg, #6366f1, #9333ea);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 9px 20px;
    font-weight: bold;
}

.stButton button:hover {
    background: linear-gradient(90deg, #4f46e5, #7e22ce);
    color: white;
}

div[data-testid="stForm"] {
    background: #172342;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #475569;
}

[data-testid="stDataFrame"] {
    border: 1px solid #475569;
    border-radius: 10px;
}

hr {
    border-color: #475569;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_data():
    if not DATASET_PATH.exists():
        return None

    df = pd.read_csv(DATASET_PATH)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    return df


@st.cache_data
def load_csv(filename):
    path = OUTPUT_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    return None


@st.cache_resource
def load_model(filename):
    path = MODEL_DIR / filename

    if path.exists():
        return joblib.load(path)

    return None


df = load_data()


# ============================================================
# 4. REUSABLE DISPLAY FUNCTIONS
# ============================================================

def metric_card(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)


def display_image(filename, caption=None):
    path = OUTPUT_DIR / filename

    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Visualization not available yet: {filename}")


def display_report(filename):
    path = OUTPUT_DIR / filename

    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            st.text(file.read())
    else:
        st.info(f"Report not available yet: {filename}")


def display_table(data, message="No results available yet."):
    if data is not None and not data.empty:
        st.dataframe(data, use_container_width=True)
    else:
        st.info(message)


def section_heading(title, description=None):
    st.markdown(f"## {title}")

    if description:
        st.caption(description)

    st.markdown("---")


def download_csv(data, filename):
    if data is not None and not data.empty:
        st.download_button(
            label=f"Download {filename}",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv"
        )


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("# 📊 CHURN LAB")
st.sidebar.caption("Customer Churn Research Dashboard")
st.sidebar.markdown("---")

pages = [
    "🏠 Overview",
    "🔍 Exploratory Analysis",
    "🤖 Model Comparison",
    "💡 Research Insights",
    "📐 Statistical Analysis",
    "🧠 Explainable AI",
    "🔄 Cross-Validation",
    "📈 Model Calibration",
    "👤 Customer Prediction",
    "⚠️ Comparative Error Analysis"
]

page = st.sidebar.radio(
    "NAVIGATION",
    pages
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Research project using the IBM Telco Customer Churn dataset."
)


# ============================================================
# 6. CHECK DATASET
# ============================================================

if df is None:
    st.error(
        "Dataset not found. Check this path:\n\n"
        "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    st.stop()


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.title("📊 Customer Churn Research")
    st.write(
        "An interactive research dashboard for customer churn "
        "analysis and machine learning model evaluation."
    )

    st.markdown("---")

    total_customers = len(df)
    churn_count = (df["Churn"] == "Yes").sum()
    retained_count = (df["Churn"] == "No").sum()
    churn_rate = churn_count / total_customers * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Customers", f"{total_customers:,}")

    with c2:
        metric_card("Churned Customers", f"{churn_count:,}")

    with c3:
        metric_card("Retained Customers", f"{retained_count:,}")

    with c4:
        metric_card("Churn Rate", f"{churn_rate:.2f}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Churn Distribution")

        churn_data = (
            df["Churn"]
            .value_counts()
            .rename_axis("Churn")
            .reset_index(name="Customers")
        )

        fig = px.pie(
            churn_data,
            names="Churn",
            values="Customers",
            hole=0.55,
            color="Churn",
            color_discrete_map={
                "Yes": "#f97316",
                "No": "#22c55e"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Customer Contract Distribution")

        contract_data = (
            df["Contract"]
            .value_counts()
            .reset_index()
        )

        contract_data.columns = ["Contract", "Customers"]

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Customers",
            color="Contract",
            text="Customers"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset Information")

    info1, info2, info3 = st.columns(3)

    info1.metric("Rows", df.shape[0])
    info2.metric("Columns", df.shape[1])
    info3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


# ============================================================
# PAGE 2: EXPLORATORY ANALYSIS
# ============================================================

elif page == "🔍 Exploratory Analysis":

    st.title("🔍 Exploratory Data Analysis")

    section_heading(
        "Understanding Customer Behaviour",
        "Explore churn patterns across customer attributes."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Churn by Contract")

        fig = px.histogram(
            df,
            x="Contract",
            color="Churn",
            barmode="group",
            color_discrete_sequence=["#22c55e", "#f97316"]
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Churn by Internet Service")

        fig = px.histogram(
            df,
            x="InternetService",
            color="Churn",
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Monthly Charges Distribution")

        fig = px.histogram(
            df,
            x="MonthlyCharges",
            color="Churn",
            marginal="box",
            opacity=0.75
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Tenure Distribution")

        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            marginal="box",
            opacity=0.75
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Additional EDA Visualizations")

    for filename in [
        "01_churn_distribution.png",
        "02_contract_churn.png",
        "03_tenure_distribution.png"
    ]:
        display_image(filename)


# ============================================================
# PAGE 3: MODEL COMPARISON
# ============================================================

elif page == "🤖 Model Comparison":

    st.title("🤖 Machine Learning Model Comparison")

    section_heading(
        "Comparative Model Evaluation",
        "Compare the performance of four classification algorithms."
    )

    results = load_csv("model_comparison.csv")

    display_table(results)

    if results is not None and not results.empty:

        st.subheader("Model Performance Comparison")

        numeric_columns = results.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            selected_metric = st.selectbox(
                "Choose evaluation metric",
                numeric_columns
            )

            model_column = next(
                (
                    col for col in results.columns
                    if col.lower() in ["model", "algorithm", "classifier"]
                ),
                results.columns[0]
            )

            fig = px.bar(
                results,
                x=model_column,
                y=selected_metric,
                color=model_column,
                text_auto=".3f"
            )

            st.plotly_chart(fig, use_container_width=True)

        download_csv(
            results,
            "model_comparison.csv"
        )

    st.subheader("ROC Curve Comparison")
    display_image("all_models_roc_comparison.png")

    st.subheader("Confusion Matrices")

    confusion_files = [
        "logistic_regression_confusion_matrix.png",
        "decision_tree_confusion_matrix.png",
        "random_forest_confusion_matrix.png",
        "gradient_boosting_confusion_matrix.png"
    ]

    for filename in confusion_files:
        display_image(filename)


# ============================================================
# PAGE 4: RESEARCH INSIGHTS
# ============================================================

elif page == "💡 Research Insights":

    st.title("💡 Research Insights")

    section_heading(
        "Feature Importance and Research Findings",
        "Explore the factors that contribute to model predictions."
    )

    st.subheader("Random Forest Feature Importance")

    display_image("random_forest_feature_importance.png")

    st.subheader("Decision Tree Feature Importance")

    display_image("decision_tree_feature_importance.png")

    st.subheader("Gradient Boosting Feature Importance")

    display_image("gradient_boosting_feature_importance.png")

    st.markdown("---")

    st.subheader("Research Results Report")

    display_report("research_results_report.txt")

    st.caption(
        "Feature importance describes model reliance on features. "
        "It does not establish causation."
    )


# ============================================================
# PAGE 5: STATISTICAL ANALYSIS
# ============================================================

elif page == "📐 Statistical Analysis":

    st.title("📐 Statistical Analysis")

    section_heading(
        "Statistical Evaluation",
        "Examine uncertainty in evaluation metrics and paired model errors."
    )

    st.subheader("Bootstrap Confidence Intervals")

    bootstrap = load_csv(
        "bootstrap_confidence_intervals.csv"
    )

    display_table(bootstrap)

    if bootstrap is not None:
        download_csv(
            bootstrap,
            "bootstrap_confidence_intervals.csv"
        )

    st.markdown("---")

    st.subheader("McNemar Statistical Test")

    mcnemar = load_csv(
        "mcnemar_test_results.csv"
    )

    display_table(mcnemar)

    if mcnemar is not None:
        download_csv(
            mcnemar,
            "mcnemar_test_results.csv"
        )

    st.markdown("---")

    st.subheader("Statistical Research Report")

    display_report(
        "statistical_analysis_report.txt"
    )

    st.info(
        "Interpret statistical significance alongside sample size, "
        "effect size, and the evaluation design."
    )


# ============================================================
# PAGE 6: EXPLAINABLE AI
# ============================================================

elif page == "🧠 Explainable AI":

    st.title("🧠 Explainable AI — SHAP Analysis")

    section_heading(
        "Understanding Model Predictions",
        "SHAP visualizations help explain how features contribute "
        "to model predictions."
    )

    st.subheader("SHAP Summary Plot")

    display_image("shap_summary.png")

    st.subheader("SHAP Feature Importance")

    display_image("shap_feature_importance.png")

    st.subheader("SHAP Waterfall Plot")

    display_image("shap_waterfall.png")

    shap_data = load_csv(
        "shap_feature_importance.csv"
    )

    st.subheader("SHAP Feature Importance Data")

    display_table(shap_data)

    if shap_data is not None:
        download_csv(
            shap_data,
            "shap_feature_importance.csv"
        )

    st.caption(
        "SHAP explanations describe the contribution of features "
        "to model predictions; they do not prove causation."
    )


# ============================================================
# PAGE 7: CROSS-VALIDATION
# ============================================================

elif page == "🔄 Cross-Validation":

    st.title("🔄 Cross-Validation Analysis")

    section_heading(
        "Five-Fold Stratified Cross-Validation",
        "Evaluate model stability across multiple data splits."
    )

    cv_results = load_csv(
        "cross_validation_results.csv"
    )

    st.subheader("Cross-Validation Results")

    display_table(cv_results)

    if cv_results is not None and not cv_results.empty:

        download_csv(
            cv_results,
            "cross_validation_results.csv"
        )

        numeric_cols = cv_results.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_cols:

            metric = st.selectbox(
                "Select cross-validation metric",
                numeric_cols,
                key="cv_metric"
            )

            fig = px.bar(
                cv_results,
                x=cv_results.columns[0],
                y=metric,
                color=cv_results.columns[0],
                text_auto=".3f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.markdown("---")

    st.subheader("Fold-Level Results")

    fold_results = load_csv(
        "cross_validation_fold_results.csv"
    )

    display_table(fold_results)

    st.markdown("---")

    st.subheader("Cross-Validation Comparison")

    display_image(
        "cross_validation_comparison.png"
    )

    st.subheader("F1 Score Stability")

    display_image(
        "cross_validation_f1_stability.png"
    )


# ============================================================
# PAGE 8: MODEL CALIBRATION
# ============================================================

elif page == "📈 Model Calibration":

    st.title("📈 Model Calibration Analysis")

    section_heading(
        "Probability Calibration",
        "Evaluate how closely predicted probabilities match observed "
        "churn frequencies."
    )

    calibration = load_csv(
        "calibration_metrics.csv"
    )

    st.subheader("Calibration Metrics")

    display_table(calibration)

    if calibration is not None and not calibration.empty:

        download_csv(
            calibration,
            "calibration_metrics.csv"
        )

        metric_candidates = [
            col for col in calibration.select_dtypes(
                include=np.number
            ).columns
            if col.lower() in ["brier score", "log loss", "brier_score",
                               "log_loss"]
        ]

        if metric_candidates:

            selected_metric = st.selectbox(
                "Choose calibration metric",
                metric_candidates
            )

            model_column = next(
                (
                    col for col in calibration.columns
                    if col.lower() in ["model", "algorithm"]
                ),
                calibration.columns[0]
            )

            fig = px.bar(
                calibration,
                x=model_column,
                y=selected_metric,
                color=model_column,
                text_auto=".4f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.markdown("---")

    st.subheader("Calibration Curve Comparison")

    display_image(
        "calibration_comparison.png"
    )

    st.info(
        "Lower Brier Score and Log Loss indicate better probability "
        "quality. A calibration curve closer to the diagonal indicates "
        "better calibration."
    )


# ============================================================
# PAGE 9: CUSTOMER PREDICTION
# ============================================================

elif page == "👤 Customer Prediction":

    st.title("👤 Customer Churn Prediction")

    section_heading(
        "Predict Customer Churn",
        "Enter customer information to receive a prediction from "
        "the trained Random Forest pipeline."
    )

    model = load_model("random_forest.pkl")

    if model is None:
        st.error(
            "Random Forest model not found. Run train.py first."
        )
        st.stop()

    st.subheader("Customer Information")

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Has Partner?",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Has Dependents?",
                ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (Months)",
                0, 72, 12
            )

        with col2:
            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col3:
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

        col4, col5 = st.columns(2)

        with col4:
            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col5:
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=2000.0,
                value=70.0
            )

            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=100000.0,
                value=float(monthly * tenure)
            )

        submitted = st.form_submit_button(
            "🔮 Predict Churn",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        try:
            prediction = model.predict(customer)[0]

            probability = None

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(customer)[0][1]

            st.markdown("---")
            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Prediction: Customer is likely to churn.")
            else:
                st.success("Prediction: Customer is likely to stay.")

            if probability is not None:
                st.metric(
                    "Predicted Churn Probability",
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(np.clip(probability, 0, 1))
                )

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )

            st.info(
                "Check that the model was trained using the same "
                "19 input features and preprocessing pipeline."
            )


# ============================================================
# PAGE 10: COMPARATIVE ERROR ANALYSIS
# ============================================================

elif page == "⚠️ Comparative Error Analysis":

    st.title("⚠️ Comparative Error Analysis")

    section_heading(
        "Understanding Model Prediction Errors",
        "Compare false positives, false negatives, overall errors, "
        "and error patterns across customer groups."
    )

    # --------------------------------------------------------
    # Overall error summary
    # --------------------------------------------------------

    st.subheader("Overall Error Summary")

    error_summary = load_csv(
        "error_analysis_summary.csv"
    )

    display_table(error_summary)

    if error_summary is not None:
        download_csv(
            error_summary,
            "error_analysis_summary.csv"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Error type comparison
    # --------------------------------------------------------

    st.subheader("False Positives and False Negatives")

    display_image(
        "error_type_comparison.png"
    )

    # --------------------------------------------------------
    # Overall error rate
    # --------------------------------------------------------

    st.subheader("Overall Model Error Rate")

    display_image(
        "overall_error_rate.png"
    )

    # --------------------------------------------------------
    # Confusion matrix comparison
    # --------------------------------------------------------

    st.subheader("Confusion Matrix Comparison")

    display_image(
        "error_confusion_comparison.png"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Customer group error analysis
    # --------------------------------------------------------

    st.subheader("Customer Group Error Analysis")

    group_errors = load_csv(
        "group_error_analysis.csv"
    )

    display_table(group_errors)

    if group_errors is not None:
        download_csv(
            group_errors,
            "group_error_analysis.csv"
        )

    display_image(
        "customer_group_error_rates.png"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Research report
    # --------------------------------------------------------

    st.subheader("Error Analysis Research Report")

    display_report(
        "error_analysis_report.txt"
    )

    st.info(
        "Group-level error differences are descriptive. They do not "
        "by themselves establish bias or explain why errors occur."
    )

    st.caption(
        "Avoid publishing individual customer identifiers or records. "
        "Use aggregate findings in research reports."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; padding:15px;">
        <h4 style="color:#a5b4fc;">
            Customer Churn Research Dashboard
        </h4>
        <p>
            Machine Learning | Statistical Analysis |
            Explainable AI | Model Evaluation
        </p>
        <p>Research and educational project</p>
    </div>
    """,
    unsafe_allow_html=True
)
