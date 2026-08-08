import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Home Credit Risk",
    page_icon="🏦",
    layout="wide"
)


# ==========================================
# LOAD MODEL & DATA
# ==========================================

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("best_model.json")
    return model


@st.cache_data
def load_features():
    feature_names = joblib.load("feature_names.pkl")
    defaults = joblib.load("feature_defaults.pkl")

    return feature_names, defaults


model = load_model()
feature_names, defaults = load_features()


# ==========================================
# TITLE
# ==========================================

st.title("🏦 Home Credit Default Risk Prediction")

st.write(
    "Predict the probability of a customer experiencing "
    "credit payment difficulties."
)

st.success(
    f"✅ Model loaded successfully | "
    f"Number of features: {len(feature_names)}"
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("📊 Customer Information")

st.sidebar.write(
    "Enter the customer's main financial information."
)


# ==========================================
# INPUT FEATURES
# ==========================================

st.subheader("👤 Customer Financial Information")

col1, col2 = st.columns(2)


with col1:

    ext_source_2 = st.slider(
        "EXT_SOURCE_2",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    ext_source_3 = st.slider(
        "EXT_SOURCE_3",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    days_id_publish = st.number_input(
        "Days ID Published",
        value=-2000,
        step=1
    )

    days_registration = st.number_input(
        "Days Registration",
        value=-3000,
        step=1
    )

    sk_id_curr = st.number_input(
        "Customer ID",
        min_value=0,
        value=100000,
        step=1
    )


with col2:

    days_last_phone_change = st.number_input(
        "Days Last Phone Change",
        value=-1000,
        step=1
    )

    employment_age_ratio = st.number_input(
        "Employment Age Ratio",
        min_value=0.0,
        value=0.5,
        step=0.01
    )

    credit_per_person = st.number_input(
        "Credit Per Person",
        min_value=0.0,
        value=100000.0,
        step=1000.0
    )

    credit_income_ratio = st.number_input(
        "Credit / Income Ratio",
        min_value=0.0,
        value=3.0,
        step=0.1
    )

    annuity_income_ratio = st.number_input(
        "Annuity / Income Ratio",
        min_value=0.0,
        value=0.2,
        step=0.01
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

predict_button = st.button(
    "🔮 Predict Default Risk",
    type="primary",
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    # Start with default values for all 100 features
    input_data = defaults.copy()

    # Replace important features with user inputs
    input_data["EXT_SOURCE_2"] = ext_source_2
    input_data["EXT_SOURCE_3"] = ext_source_3
    input_data["DAYS_ID_PUBLISH"] = days_id_publish
    input_data["DAYS_REGISTRATION"] = days_registration
    input_data["SK_ID_CURR"] = sk_id_curr
    input_data["DAYS_LAST_PHONE_CHANGE"] = days_last_phone_change
    input_data["EMPLOYMENT_AGE_RATIO"] = employment_age_ratio
    input_data["CREDIT_PER_PERSON"] = credit_per_person
    input_data["CREDIT_INCOME_RATIO"] = credit_income_ratio
    input_data["ANNUITY_INCOME_RATIO"] = annuity_income_ratio

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make sure feature order is exactly the same
    input_df = input_df[feature_names]

    # Prediction probability
    probability = model.predict_proba(input_df)[0][1]

    prediction = 1 if probability >= 0.5 else 0

    probability_percent = probability * 100


    # ==========================================
    # RESULTS
    # ==========================================

    st.subheader("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Default Probability",
            f"{probability_percent:.2f}%"
        )


    with result_col2:

        if prediction == 1:

            st.error(
                "🔴 HIGH RISK\n\n"
                "The model predicts a higher risk of "
                "payment difficulties."
            )

        else:

            st.success(
                "🟢 LOW RISK\n\n"
                "The model predicts a lower risk of "
                "payment difficulties."
            )


    # Progress bar
    st.progress(float(probability))


    # Explanation
    st.info(
        "The prediction is generated using the trained "
        "XGBoost model and the same 100 features used during training."
    )
    