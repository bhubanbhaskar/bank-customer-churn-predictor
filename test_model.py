import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# Load model and preprocessors
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.h5")

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("onehot_encoder_geography.pkl", "rb") as f:
        onehot_encoder_geography = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geography, scaler


model, label_encoder_gender, onehot_encoder_geography, scaler = load_artifacts()

# App UI
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")

# Inputs
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
    "Tenure",
    min_value=0,
    max_value=10,
    value=3
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=60000.0
)

num_of_products = st.number_input(
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

# Prediction
if st.button("Predict Churn"):

    try:
        st.write("Button clicked")

        input_data = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary
        }

        input_df = pd.DataFrame([input_data])

        # Encode gender
        input_df["Gender"] = label_encoder_gender.transform(
            input_df["Gender"]
        )

        # One-hot encode geography
        geo_encoded = onehot_encoder_geography.transform(
            input_df[["Geography"]]
        )

        geo_df = pd.DataFrame(
            geo_encoded.toarray(),
            columns=onehot_encoder_geography.get_feature_names_out(
                ["Geography"]
            )
        )

        # Combine
        input_df = pd.concat(
            [input_df.drop("Geography", axis=1), geo_df],
            axis=1
        )

        st.subheader("Processed Input")
        st.dataframe(input_df)

        st.write("Shape:", input_df.shape)

        # Scale
        input_scaled = scaler.transform(input_df)

        st.write("Scaling completed")
        st.write("Scaled shape:", input_scaled.shape)

        # Predict
        st.write("Running prediction...")

        prediction = model.predict(
            input_scaled,
            verbose=0
        )

        st.write("Prediction finished")

        probability = float(prediction[0][0])
        predicted_class = 1 if probability > 0.5 else 0

        st.success(
            f"Predicted Probability of Churn: {probability:.4f}"
        )

        if predicted_class == 1:
            st.error("Customer is likely to churn")
        else:
            st.success("Customer is likely to stay")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)