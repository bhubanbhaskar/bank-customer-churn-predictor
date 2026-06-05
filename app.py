import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import sys

st.write("Python executable:")
st.write(sys.executable)

st.write("TensorFlow version:")
st.write(tf.__version__)

st.write("TensorFlow path:")
st.write(tf.__file__)

# Load model and preprocessors
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geography.pkl", "rb") as f:
    onehot_encoder_geography = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict customer churn.")

# User Inputs
credit_score = st.number_input("Credit Score", 300, 850, 600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 40)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, value=60000.0)
num_products = st.number_input("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0.0, value=50000.0)

if st.button("Predict Churn"):

    try:
        # Create input dataframe
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

        st.subheader("Original Input")
        st.dataframe(input_df)

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

        # Merge encoded geography
        input_df = pd.concat(
            [input_df.drop("Geography", axis=1), geography_df],
            axis=1
        )

        st.subheader("Processed Input")
        st.dataframe(input_df)

        # Reorder columns exactly as scaler expects
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

        st.write("Shape:", input_df.shape)
        st.write("Model input shape:", model.input_shape)

        # Scale
        input_scaled = scaler.transform(input_df)

        st.write("Scaling completed")
        st.write("Scaled shape:", input_scaled.shape)

        # Convert to float32
        input_scaled = np.asarray(input_scaled, dtype=np.float32)

        st.write("Contains NaN:", np.isnan(input_scaled).any())
        st.write("Contains Inf:", np.isinf(input_scaled).any())
        st.write("Dtype:", input_scaled.dtype)
        

        

        # Prediction
        prediction = model.predict(input_scaled, verbose=0)

        st.write("Prediction completed")

        probability = float(prediction[0][0])

        st.success(
            f"Probability of Churn: {probability:.2%}"
        )

        if probability > 0.5:
            st.error("Prediction: Customer is likely to churn.")
        else:
            st.success("Prediction: Customer is likely to stay.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)