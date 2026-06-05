import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense   
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime  

## Load the dataset
data = pd.read_csv('Churn_Modelling.csv')
#print(data.head())

## Preprocess the data
# Drop unnecessary columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
#print(data.head())
# Encode categorical variables
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
#print(data.head())

## one-hot encode the 'Geography' column
onehot_encoder_geography = OneHotEncoder()
geography_encoded = onehot_encoder_geography.fit_transform(data[['Geography']])
geography_df = pd.DataFrame(geography_encoded.toarray(), columns=onehot_encoder_geography.get_feature_names_out(['Geography']))
data = pd.concat([data, geography_df], axis=1)
data = data.drop('Geography', axis=1)
#print(data.head())
## save the encoder and scaler for later use
with open('label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(label_encoder_gender, f)
with open('onehot_encoder_geography.pkl', 'wb') as f:
    pickle.dump(onehot_encoder_geography, f)

## divide the data into dependent and independent features
X = data.drop('Exited', axis=1)
y = data['Exited']
## Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
## save the scaler for later use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

    ## Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
## Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
## set up tensorboard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
## set up early stopping callback
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
## Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping_callback, tensorboard_callback])

## Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

## Save the model
model.save('model.h5')