import json
import joblib
import pandas as pd

from pathlib import Path

from catboost import CatBoostClassifier, Pool


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_PATH = (
    BASE_DIR
    / "models"
    / "catboost_churn_model.cbm"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "models"
    / "catboost_threshold.joblib"
)

SCHEMA_PATH = (
    BASE_DIR
    / "models"
    / "feature_schema.json"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / "model_metadata.json"
)

FEATURE_METADATA_PATH = (
    BASE_DIR
    / "models"
    / "feature_metadata.json"
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    model = CatBoostClassifier()

    model.load_model(
        str(MODEL_PATH)
    )

    return model


# ============================================================
# THRESHOLD
# ============================================================

def load_threshold():

    return joblib.load(
        THRESHOLD_PATH
    )


# ============================================================
# FEATURE SCHEMA
# ============================================================

def load_schema():

    with open(
        SCHEMA_PATH,
        "r"
    ) as file:

        return json.load(file)


# ============================================================
# MODEL METADATA
# ============================================================

def load_metadata():

    with open(
        METADATA_PATH,
        "r"
    ) as file:

        return json.load(file)


# ============================================================
# FEATURE METADATA
# ============================================================

def load_feature_metadata():

    with open(
        FEATURE_METADATA_PATH,
        "r"
    ) as file:

        return json.load(file)


# ============================================================
# RISK ASSIGNMENT
# ============================================================

def assign_risk(probability):

    if probability < 0.30:

        return "Low Risk"

    elif probability < 0.60:

        return "Medium Risk"

    return "High Risk"


# ============================================================
# RETENTION ACTION
# ============================================================

def get_recommendation(risk):

    if risk == "High Risk":

        return "Immediate retention intervention"

    elif risk == "Medium Risk":

        return "Proactive engagement"

    return "Routine engagement"


# ============================================================
# DATAFRAME PREDICTION
# ============================================================

def predict_dataframe(df):

    model = load_model()

    threshold = load_threshold()

    probability = (
        model.predict_proba(df)[:, 1]
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    result = df.copy()

    result["Churn_Probability"] = probability

    result["Churn_Prediction"] = prediction

    result["Risk_Level"] = [
        assign_risk(p)
        for p in probability
    ]

    result["Recommended_Action"] = [
        get_recommendation(r)
        for r in result["Risk_Level"]
    ]

    return result


# ============================================================
# SINGLE CUSTOMER PREDICTION
# ============================================================

def predict_single(df):

    result = predict_dataframe(df)

    return result.iloc[0]


# ============================================================
# SHAP EXPLANATION
# ============================================================

def explain_prediction(df):

    model = load_model()

    feature_metadata = load_feature_metadata()

    categorical_features = feature_metadata.get(
        "categorical_features",
        []
    )

    # Make sure categorical columns are strings
    df_explain = df.copy()

    for column in categorical_features:

        if column in df_explain.columns:

            df_explain[column] = (
                df_explain[column]
                .astype(str)
            )

    # Create CatBoost Pool
    explanation_pool = Pool(
        data=df_explain,
        cat_features=categorical_features
    )

    # CatBoost native SHAP values
    shap_values = model.get_feature_importance(
        explanation_pool,
        type="ShapValues"
    )

    # Remove expected value column
    feature_shap_values = shap_values[
        :, :-1
    ]

    expected_values = shap_values[
        :, -1
    ]

    explanation = pd.DataFrame(
        {
            "Feature": df_explain.columns,
            "SHAP_Value": feature_shap_values[0],
            "Feature_Value": [
                df_explain.iloc[0][column]
                for column in df_explain.columns
            ]
        }
    )

    explanation["Impact"] = (
        explanation["SHAP_Value"]
        .abs()
    )

    explanation["Direction"] = (
        explanation["SHAP_Value"]
        .apply(
            lambda x:
            "Increases Churn Risk"
            if x > 0
            else "Decreases Churn Risk"
        )
    )

    explanation = (
        explanation
        .sort_values(
            "Impact",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return explanation, expected_values[0]

# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():

    model = load_model()

    feature_importance = model.get_feature_importance(
        prettified=True
    )

    feature_importance = (
        feature_importance
        .rename(
            columns={
                "Feature Id": "Feature",
                "Importances": "Importance"
            }
        )
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return feature_importance

# ============================================================
# MODEL PERFORMANCE
# ============================================================

def load_performance():

    performance_path = (
        BASE_DIR
        / "models"
        / "model_performance.json"
    )

    with open(
        performance_path,
        "r"
    ) as file:

        return json.load(file)

# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input_dataframe(df):

    model = load_model()

    required_features = list(
        model.feature_names_
    )

    # --------------------------------------------------------
    # Empty dataframe
    # --------------------------------------------------------

    if df is None or df.empty:

        return {
            "valid": False,
            "message": "The uploaded dataset is empty.",
            "data": None
        }

    # --------------------------------------------------------
    # Missing columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in required_features
        if column not in df.columns
    ]

    if missing_columns:

        return {
            "valid": False,
            "message": (
                "Missing required columns: "
                + ", ".join(missing_columns)
            ),
            "data": None
        }

    # --------------------------------------------------------
    # Keep only model features
    # --------------------------------------------------------

    cleaned_df = df[
        required_features
    ].copy()

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = (
        cleaned_df
        .isnull()
        .sum()
        .sum()
    )

    if missing_values > 0:

        return {
            "valid": False,
            "message": (
                f"The dataset contains "
                f"{missing_values} missing values."
            ),
            "data": None
        }

    # --------------------------------------------------------
    # Return cleaned dataframe
    # --------------------------------------------------------

    return {
        "valid": True,
        "message": "Input validation successful.",
        "data": cleaned_df
    }