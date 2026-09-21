import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

sys.path.append(str(BASE_DIR))

from src.predictor import (
    predict_dataframe,
    predict_single,
    explain_prediction,
    get_feature_importance,
    validate_input_dataframe,
    load_threshold,
    load_schema,
    load_metadata,
    load_performance
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    /* KPI cards */
    .metric-card {
        padding: 22px;
        border-radius: 14px;
        background: #1f2937;
        border: 1px solid #374151;
        min-height: 130px;
    }

    .metric-label {
        font-size: 15px;
        color: #9ca3af;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
    }

    /* Risk cards */
    .risk-card {
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #374151;
    }

    .risk-title {
        font-size: 15px;
        color: #9ca3af;
    }

    .risk-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Prediction box */
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        background: #1f2937;
        border: 1px solid #374151;
        margin-top: 15px;
    }

    .prediction-title {
        font-size: 18px;
        color: #9ca3af;
    }

    .prediction-value {
        font-size: 38px;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #374151;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def get_model():
    from src.predictor import load_model
    return load_model()


@st.cache_data
def get_schema():
    return load_schema()


@st.cache_data
def get_metadata():
    return load_metadata()

@st.cache_data
def get_performance():
    return load_performance()

model = get_model()
schema = get_schema()
metadata = get_metadata()
threshold = load_threshold()
performance = get_performance()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Customer Churn Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered customer churn prediction '
    'and retention prioritization platform.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Navigation")

    st.caption(
        "Customer Churn Intelligence Platform"
    )

    page = st.radio(
        "Select Module",
        [
            "Executive Dashboard",
            "Single Customer",
            "Batch Prediction",
            "Explainability",
            "Model Analytics",
            "Model Information"
        ]
    )

    st.divider()

    st.caption(
        f"Model: {metadata.get('model_name', 'CatBoost')}"
    )

    st.caption(
        f"Threshold: {threshold:.3f}"
    )


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.header(
        "Executive Dashboard"
    )

    st.write(
        "Upload customer data to generate "
        "churn intelligence and retention priorities."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
        key="dashboard_upload"
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to start the churn analysis."
        )

    else:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            required_columns = list(
                schema.keys()
            )

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                st.error(
                    "The uploaded file is missing required features."
                )

                st.write(
                    missing_columns
                )

            else:

                prediction_input = df[
                    required_columns
                ].copy()

                predictions = predict_dataframe(
                    prediction_input
                )

                total_customers = len(
                    predictions
                )

                predicted_churners = int(
                    predictions[
                        "Churn_Prediction"
                    ].sum()
                )

                high_risk = int(
                    (
                        predictions[
                            "Risk_Level"
                        ] == "High Risk"
                    ).sum()
                )

                avg_probability = float(
                    predictions[
                        "Churn_Probability"
                    ].mean()
                )

                # ------------------------------
                # KPI CARDS
                # ------------------------------

                st.subheader(
                    "Key Performance Indicators"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                Total Customers
                            </div>
                            <div class="metric-value">
                                {total_customers:,}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                Predicted Churners
                            </div>
                            <div class="metric-value">
                                {predicted_churners:,}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col3:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                High Risk Customers
                            </div>
                            <div class="metric-value">
                                {high_risk:,}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col4:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                Average Churn Probability
                            </div>
                            <div class="metric-value">
                                {avg_probability:.1%}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.divider()

                # ------------------------------
                # RISK DISTRIBUTION
                # ------------------------------

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader(
                        "Risk Distribution"
                    )

                    risk_counts = (
                        predictions[
                            "Risk_Level"
                        ]
                        .value_counts()
                        .reindex(
                            [
                                "Low Risk",
                                "Medium Risk",
                                "High Risk"
                            ],
                            fill_value=0
                        )
                        .reset_index()
                    )

                    risk_counts.columns = [
                        "Risk_Level",
                        "Customers"
                    ]

                    fig = px.bar(
                        risk_counts,
                        x="Risk_Level",
                        y="Customers",
                        title="Customers by Risk Level",
                        text="Customers"
                    )

                    fig.update_layout(
                        showlegend=False
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                with col2:

                    st.subheader(
                        "Churn Probability Distribution"
                    )

                    fig = px.histogram(
                        predictions,
                        x="Churn_Probability",
                        nbins=20,
                        title="Predicted Churn Probability"
                    )

                    fig.update_layout(
                        xaxis_title="Churn Probability",
                        yaxis_title="Customers"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                st.divider()

                # ------------------------------
                # HIGH RISK TABLE
                # ------------------------------

                st.subheader(
                    "🚨 Highest Risk Customers"
                )

                high_risk_customers = (
                    predictions
                    .sort_values(
                        "Churn_Probability",
                        ascending=False
                    )
                    .head(20)
                )

                st.dataframe(
                    high_risk_customers,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------
                # DOWNLOAD
                # ------------------------------

                st.divider()

                csv_data = predictions.to_csv(
                    index=False
                )

                st.download_button(
                    "⬇Download Complete Predictions",
                    data=csv_data,
                    file_name="customer_churn_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# SINGLE CUSTOMER
# ============================================================

elif page == "Single Customer":

    st.header(
        "Single Customer Prediction"
    )

    st.write(
        "Enter customer information to estimate "
        "their churn probability."
    )

    st.divider()

    user_data = {}

    columns = list(
        schema.keys()
    )

    # Divide inputs into columns
    input_columns = st.columns(2)

    for index, column in enumerate(columns):

        info = schema[column]

        with input_columns[index % 2]:

            st.markdown(
                f"**{column}**"
            )

            if info["type"] == "categorical":

                options = info.get(
                    "values",
                    []
                )

                default = info.get(
                    "default"
                )

                if default in options:
                    default_index = options.index(
                        default
                    )
                else:
                    default_index = 0

                user_data[column] = st.selectbox(
                    column,
                    options,
                    index=default_index,
                    label_visibility="collapsed"
                )

            else:

                minimum = info["min"]
                maximum = info["max"]
                default = info["default"]

                user_data[column] = st.number_input(
                    column,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default),
                    label_visibility="collapsed"
                )

    st.divider()

    if st.button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [user_data]
        )

        try:

            # Save the latest customer for explainability
            st.session_state["last_customer"] = input_df.copy()

            validation = validate_input_dataframe(
                input_df
            )

            if not validation["valid"]:

                st.error(
                    validation["message"]
                )

                st.stop()

            input_df = validation["data"]

            result = predict_single(
                input_df
            )

            probability = float(
                result[
                    "Churn_Probability"
                ]
            )

            risk = result[
                "Risk_Level"
            ]

            action = result[
                "Recommended_Action"
            ]

            prediction = int(
                result[
                    "Churn_Prediction"
                ]
            )

            st.success(
                "Prediction generated successfully."
            )

            st.divider()

            # ------------------------------
            # RESULT CARDS
            # ------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Churn Probability",
                    f"{probability:.1%}"
                )

            with col2:

                st.metric(
                    "Risk Level",
                    risk
                )

            with col3:

                st.metric(
                    "Prediction",
                    "Churn"
                    if prediction == 1
                    else "No Churn"
                )

            # ------------------------------
            # GAUGE
            # ------------------------------

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={
                        "text":
                        "Churn Probability"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value":
                            threshold * 100
                        }
                    }
                )
            )

            gauge.update_layout(
                height=350
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            # ------------------------------
            # RECOMMENDATION
            # ------------------------------

            st.subheader(
                "Recommended Action"
            )

            if risk == "High Risk":

                st.error(
                    f"{action}"
                )

            elif risk == "Medium Risk":

                st.warning(
                    f"{action}"
                )

            else:

                st.success(
                    f"{action}"
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    st.header(
        "Batch Customer Prediction"
    )

    st.write(
        "Upload a CSV containing the customer "
        "features required by the model."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_upload"
    )

    if uploaded_file is not None:

        try:

            # ==================================================
            # READ CSV
            # ==================================================

            df = pd.read_csv(
                uploaded_file
            )

            # ==================================================
            # INPUT VALIDATION
            # ==================================================

            validation = validate_input_dataframe(
                df
            )

            if not validation["valid"]:

                st.error(
                    validation["message"]
                )

                st.stop()

            # Use validated / cleaned dataframe
            validated_df = validation["data"]

            # ==================================================
            # PREDICTION
            # ==================================================

            predictions = predict_dataframe(
                validated_df
            )

            st.success(
                "Batch prediction completed successfully."
            )

            # ==================================================
            # KPI
            # ==================================================

            total = len(
                predictions
            )

            churners = int(
                predictions[
                    "Churn_Prediction"
                ].sum()
            )

            high_risk = int(
                (
                    predictions[
                        "Risk_Level"
                    ] == "High Risk"
                ).sum()
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Customers",
                total
            )

            col2.metric(
                "Predicted Churners",
                churners
            )

            col3.metric(
                "High Risk",
                high_risk
            )

            st.divider()

            # ==================================================
            # RESULTS TABLE
            # ==================================================

            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                predictions,
                use_container_width=True,
                hide_index=True
            )

            # ==================================================
            # DOWNLOAD RESULTS
            # ==================================================

            csv_data = predictions.to_csv(
                index=False
            )

            st.download_button(
                "Download Prediction Results",
                data=csv_data,
                file_name=(
                    "customer_churn_predictions.csv"
                ),
                mime="text/csv",
                type="primary"
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.header(
        "Customer Prediction Explainability"
    )

    st.write(
        """
        Understand which customer attributes are
        contributing to the predicted churn risk.
        """
    )

    # --------------------------------------------------------
    # CHECK WHETHER A CUSTOMER HAS BEEN PREDICTED
    # --------------------------------------------------------

    if "last_customer" not in st.session_state:

        st.info(
            "No customer prediction is available yet."
        )

        st.markdown(
            """
            ### How to use this section

            1. Go to **Single Customer**
            2. Enter customer information
            3. Click **Predict Customer Churn**
            4. Return to **Explainability**
            """
        )

    else:

        customer_df = (
            st.session_state["last_customer"]
        )

        # ----------------------------------------------------
        # CUSTOMER PREDICTION
        # ----------------------------------------------------

        result = predict_single(
            customer_df
        )

        probability = float(
            result["Churn_Probability"]
        )

        risk = result[
            "Risk_Level"
        ]

        prediction = int(
            result["Churn_Prediction"]
        )

        # ----------------------------------------------------
        # PREDICTION SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Prediction Summary"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Churn Probability",
                f"{probability:.1%}"
            )

        with col2:

            st.metric(
                "Risk Level",
                risk
            )

        with col3:

            st.metric(
                "Prediction",
                "Churn"
                if prediction == 1
                else "No Churn"
            )

        st.divider()

        # ----------------------------------------------------
        # SHAP CALCULATION
        # ----------------------------------------------------

        try:

            explanation, base_value = (
                explain_prediction(
                    customer_df
                )
            )

            # ------------------------------------------------
            # TOP FEATURES
            # ------------------------------------------------

            st.subheader(
                "Top Factors Influencing This Prediction"
            )

            st.caption(
                "Features are ranked by the magnitude "
                "of their SHAP contribution."
            )

            top_features = (
                explanation
                .head(10)
                .copy()
            )

            # ------------------------------------------------
            # FORMAT DISPLAY TABLE
            # ------------------------------------------------

            display_table = top_features[
                [
                    "Feature",
                    "Feature_Value",
                    "SHAP_Value",
                    "Direction"
                ]
            ].copy()

            display_table[
                "SHAP_Value"
            ] = display_table[
                "SHAP_Value"
            ].round(4)

            display_table.columns = [
                "Feature",
                "Customer Value",
                "SHAP Contribution",
                "Impact on Churn"
            ]

            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # ------------------------------------------------
            # POSITIVE / NEGATIVE SHAP
            # ------------------------------------------------

            positive_features = (
                explanation[
                    explanation[
                        "SHAP_Value"
                    ] > 0
                ]
                .head(5)
            )

            negative_features = (
                explanation[
                    explanation[
                        "SHAP_Value"
                    ] < 0
                ]
                .head(5)
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Factors Increasing Churn Risk"
                )

                if len(
                    positive_features
                ) == 0:

                    st.success(
                        "No positive SHAP contributions found."
                    )

                else:

                    for _, row in (
                        positive_features.iterrows()
                    ):

                        st.markdown(
                            f"""
                            **{row['Feature']}**

                            Customer value:
                            `{row['Feature_Value']}`

                            SHAP contribution:
                            `{row['SHAP_Value']:.4f}`
                            """
                        )

                        st.divider()

            with col2:

                st.subheader(
                    "Factors Decreasing Churn Risk"
                )

                if len(
                    negative_features
                ) == 0:

                    st.info(
                        "No negative SHAP contributions found."
                    )

                else:

                    for _, row in (
                        negative_features.iterrows()
                    ):

                        st.markdown(
                            f"""
                            **{row['Feature']}**

                            Customer value:
                            `{row['Feature_Value']}`

                            SHAP contribution:
                            `{row['SHAP_Value']:.4f}`
                            """
                        )

                        st.divider()

            # ------------------------------------------------
            # SHAP BAR CHART
            # ------------------------------------------------

            st.subheader(
                "Feature Contribution Chart"
            )

            chart_data = (
                explanation
                .head(10)
                .sort_values(
                    "SHAP_Value"
                )
            )

            fig = px.bar(
                chart_data,
                x="SHAP_Value",
                y="Feature",
                orientation="h",
                title=(
                    "Top Feature Contributions "
                    "to Churn Prediction"
                ),
                hover_data=[
                    "Feature_Value",
                    "Direction"
                ]
            )

            fig.add_vline(
                x=0,
                line_dash="dash"
            )

            fig.update_layout(
                height=500,
                xaxis_title="SHAP Contribution",
                yaxis_title="Feature"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ------------------------------------------------
            # BUSINESS INTERPRETATION
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "Business Interpretation"
            )

            strongest_positive = (
                positive_features.iloc[0]
                if len(
                    positive_features
                ) > 0
                else None
            )

            strongest_negative = (
                negative_features.iloc[0]
                if len(
                    negative_features
                ) > 0
                else None
            )

            if strongest_positive is not None:

                st.warning(
                    f"The strongest factor increasing "
                    f"this customer's churn risk is "
                    f"**{strongest_positive['Feature']}** "
                    f"with a SHAP contribution of "
                    f"**{strongest_positive['SHAP_Value']:.4f}**."
                )

            if strongest_negative is not None:

                st.success(
                    f"The strongest factor decreasing "
                    f"this customer's churn risk is "
                    f"**{strongest_negative['Feature']}** "
                    f"with a SHAP contribution of "
                    f"**{strongest_negative['SHAP_Value']:.4f}**."
                )

            st.info(
                """
                SHAP values describe the contribution of
                individual features to the model prediction.
                A positive contribution pushes the prediction
                toward higher churn risk, while a negative
                contribution pushes it toward lower churn risk.
                """
            )

        except Exception as e:

            st.error(
                f"Explainability analysis failed: {e}"
            )


# ============================================================
# MODEL ANALYTICS
# ============================================================

elif page == "Model Analytics":

    st.header(
        "📈 Model Analytics"
    )

    st.write(
        "Performance evaluation and global model insights."
    )

    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    st.subheader(
        "Final Model Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            f"{performance['accuracy']:.4f}"
        )

    with col2:

        st.metric(
            "Precision",
            f"{performance['precision']:.4f}"
        )

    with col3:

        st.metric(
            "Recall",
            f"{performance['recall']:.4f}"
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "F1 Score",
            f"{performance['f1']:.4f}"
        )

    with col5:

        st.metric(
            "ROC-AUC",
            f"{performance['roc_auc']:.4f}"
        )

    with col6:

        st.metric(
            "PR-AUC",
            f"{performance['pr_auc']:.4f}"
        )

    st.divider()

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "Model Selection"
    )

    selection_data = pd.DataFrame({
        "Property": [
            "Final Model",
            "Primary Metric",
            "Selection Method",
            "Threshold Method",
            "Decision Threshold"
        ],
        "Value": [
            performance["model"],
            metadata.get(
                "primary_metric",
                "PR-AUC"
            ),
            metadata.get(
                "selection_method",
                "OOF PR-AUC"
            ),
            metadata.get(
                "threshold_selection",
                "OOF F1 optimization"
            ),
            f"{performance['threshold']:.3f}"
        ]
    })

    st.dataframe(
        selection_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # GLOBAL FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "🌐 Global Feature Importance"
    )

    st.write(
        """
        This shows which features the CatBoost model
        considers important across the overall dataset.
        """
    )

    try:

        importance = (
            get_feature_importance()
        )

        top_features = (
            importance
            .head(15)
            .sort_values(
                "Importance"
            )
        )

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Global Feature Importance"
        )

        fig.update_layout(
            height=600,
            xaxis_title="Importance",
            yaxis_title="Feature"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            importance.head(15),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(
            f"Feature importance failed: {e}"
        )

    st.divider()

    # ========================================================
    # MODEL PIPELINE
    # ========================================================

    st.subheader(
        "Machine Learning Pipeline"
    )

    st.code(
        """
Customer Data
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
CatBoost Classifier
      ↓
Churn Probability
      ↓
OOF-Optimized Threshold
      ↓
Churn Prediction
      ↓
Risk Segmentation
      ↓
Retention Recommendation
      ↓
SHAP Explainability
        """,
        language="text"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.header(
        "Model Information"
    )

    st.write(
        """
        Technical information about the deployed customer
        churn prediction model, training strategy,
        validation methodology and decision threshold.
        """
    )

    # ========================================================
    # MODEL OVERVIEW
    # ========================================================

    st.subheader(
        "Model Overview"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            performance["model"]
        )

    with col2:

        st.metric(
            "Primary Metric",
            "PR-AUC"
        )

    with col3:

        st.metric(
            "Decision Threshold",
            f"{performance['threshold']:.3f}"
        )

    st.divider()

    # ========================================================
    # MODEL SELECTION
    # ========================================================

    st.subheader(
        "Model Selection Strategy"
    )

    selection_info = pd.DataFrame(
        {
            "Component": [
                "Candidate Models",
                "Final Model",
                "Primary Selection Metric",
                "Selection Strategy",
                "Threshold Strategy"
            ],
            "Description": [
                "Logistic Regression, Random Forest, XGBoost, CatBoost",
                "CatBoost",
                "PR-AUC",
                "Out-of-Fold PR-AUC comparison",
                "Out-of-Fold F1 optimization"
            ]
        }
    )

    st.dataframe(
        selection_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # VALIDATION STRATEGY
    # ========================================================

    st.subheader(
        "Validation Strategy"
    )

    st.markdown(
        """
        **Out-of-Fold (OOF) evaluation** is used to reduce
        optimistic performance estimation during model
        comparison and threshold selection.

        The final decision threshold is selected using
        out-of-fold predictions rather than the training
        predictions of the fitted model.
        """
    )

    st.divider()

    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.subheader(
        "Final Model Performance"
    )

    performance_table = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC"
            ],
            "Score": [
                performance["accuracy"],
                performance["precision"],
                performance["recall"],
                performance["f1"],
                performance["roc_auc"],
                performance["pr_auc"]
            ]
        }
    )

    performance_table["Score"] = (
        performance_table["Score"]
        .round(4)
    )

    st.dataframe(
        performance_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    st.subheader(
        "Explainability"
    )

    st.markdown(
        """
        The application uses **CatBoost native SHAP values**
        for local prediction explanations.

        SHAP explanations are used to identify:

        - Features increasing churn risk
        - Features decreasing churn risk
        - Relative contribution of individual features
        - Customer-specific prediction drivers
        """
    )

    st.divider()

    # ========================================================
    # DECISION LOGIC
    # ========================================================

    st.subheader(
        "Risk Segmentation Logic"
    )

    risk_table = pd.DataFrame(
        {
            "Churn Probability": [
                "< 30%",
                "30% – < 60%",
                "≥ 60%"
            ],
            "Risk Level": [
                "Low Risk",
                "Medium Risk",
                "High Risk"
            ],
            "Recommended Action": [
                "Routine engagement",
                "Proactive engagement",
                "Immediate retention intervention"
            ]
        }
    )

    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # FEATURE INFORMATION
    # ========================================================

    st.subheader(
        "Feature Information"
    )

    try:

        feature_metadata = load_feature_metadata()

        categorical_features = (
            feature_metadata.get(
                "categorical_features",
                []
            )
        )

        numerical_features = (
            feature_metadata.get(
                "numerical_features",
                []
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### Categorical Features"
            )

            st.write(
                categorical_features
            )

        with col2:

            st.markdown(
                "### Numerical Features"
            )

            st.write(
                numerical_features
            )

    except Exception as e:

        st.info(
            "Feature metadata is not available."
        )

    st.divider()

    # ========================================================
    # PIPELINE
    # ========================================================

    st.subheader(
        "End-to-End ML Pipeline"
    )

    st.code(
        """
Customer Data
      │
      ▼
Input Validation
      │
      ▼
Feature Engineering
      │
      ▼
Candidate Model Evaluation
      │
      ▼
OOF Model Comparison
      │
      ▼
CatBoost Selection
      │
      ▼
OOF Threshold Optimization
      │
      ▼
Churn Probability
      │
      ▼
Risk Segmentation
      │
      ▼
Retention Prioritization
      │
      ▼
SHAP Explainability
        """,
        language="text"
    )