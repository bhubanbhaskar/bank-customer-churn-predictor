### load the trained model, scaler pickle, onehot encoder pickle, and label encoder pickle
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  

model = load_model('model.h5')

## load the encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geography = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
## Example input data for prediction
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
## Preprocess the input data
input_df = pd.DataFrame([input_data])
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
geography_encoded = onehot_encoder_geography.transform(input_df[['Geography']])
geography_df = pd.DataFrame(geography_encoded.toarray(), columns=onehot_encoder_geography.get_feature_names_out(['Geography']))
input_df = pd.concat([input_df, geography_df], axis=1)
input_df = input_df.drop('Geography', axis=1)
input_scaled = scaler.transform(input_df)
## Make a prediction
prediction = model.predict(input_scaled)
predicted_class = (prediction > 0.5).astype(int)
print(f'Predicted Probability of Churn: {prediction[0][0]:.4f}')
print(f'Predicted Class (0 = No Churn, 1 = Churn): {predicted_class[0][0]}')    
