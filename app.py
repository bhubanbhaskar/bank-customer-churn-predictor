import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

# --------------------------------------------------
# Load Model and Preprocessors
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_resource
def load_objects():
    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("onehot_encoder_geography.pkl", "rb") as f:
        onehot_encoder_geography = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return label_encoder_gender, onehot_encoder_geography, scaler


model = load_model()
label_encoder_gender, onehot_encoder_geography, scaler = load_objects()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🏦 Bank Customer Churn Predictor")
st.markdown(
    "Predict whether a customer is likely to leave the bank based on customer details."
)

st.divider()

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=600
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

tenure = st.number_input(
    "Tenure (Years)",
    min_value=0,
    max_value=10,
    value=3
)

balance = st.number_input(
    "Account Balance",
    min_value=0.0,
    value=60000.0
)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Churn", use_container_width=True):

    try:
        # Create DataFrame
        input_data = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary
        }

        input_df = pd.DataFrame([input_data])

        # Encode Gender
        input_df["Gender"] = label_encoder_gender.transform(
            input_df["Gender"]
        )

        # Encode Geography
        geography_encoded = onehot_encoder_geography.transform(
            input_df[["Geography"]]
        )

        geography_df = pd.DataFrame(
            geography_encoded.toarray(),
            columns=onehot_encoder_geography.get_feature_names_out(
                ["Geography"]
            )
        )

        # Merge geography columns
        input_df = pd.concat(
            [input_df.drop("Geography", axis=1), geography_df],
            axis=1
        )

        # Correct column order
        expected_columns = [
            "CreditScore",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Geography_France",
            "Geography_Germany",
            "Geography_Spain"
        ]

        input_df = input_df[expected_columns]

        # Scale
        input_scaled = scaler.transform(input_df)
        input_scaled = np.asarray(input_scaled, dtype=np.float32)

        # Predict
        prediction = model.predict(input_scaled, verbose=0)

        probability = float(prediction[0][0])

        # --------------------------------------------------
        # Results
        # --------------------------------------------------
        st.divider()
        st.subheader("📊 Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

        st.progress(int(probability * 100))

        if probability < 0.30:
            st.success("🟢 Risk Level: LOW")

        elif probability < 0.70:
            st.warning("🟠 Risk Level: MEDIUM")

        else:
            st.error("🔴 Risk Level: HIGH")

        if probability > 0.5:
            st.error(
                f"⚠️ Customer is likely to churn ({probability:.2%})"
            )
        else:
            st.success(
                f"✅ Customer is likely to stay ({probability:.2%})"
            )

        # Optional debug section
        with st.expander("View Processed Input"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

    st.divider()

st.caption("Developed by Bhuban Bhaskar 🚀 | Bank Customer Churn Predictor")