import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import os

# Define Crop Dictionary
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7, 'apple': 8,
    'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Load or train model
MODEL_PATH = "pickle/crop_model.pkl"
SCALER_PATH = "pickle/crop_scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("Loading saved crop model...")
    rfc = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
else:
    print("Training new crop model...")
    # Load dataset
    crop = pd.read_csv("data/Crop_rec.csv")
    
    # Preprocessing
    crop['crop_num'] = crop['label'].map(crop_dict)
    crop.drop(['label'], axis=1, inplace=True)
    X, y = crop.drop(['crop_num'], axis=1), crop['crop_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest model
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    # Save the model and scaler
    os.makedirs("pickle", exist_ok=True)
    pickle.dump(rfc, open(MODEL_PATH, "wb"))
    pickle.dump(scaler, open(SCALER_PATH, "wb"))
    print("Crop model trained and saved!")

# Prediction function
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)
    prediction = rfc.predict(features_scaled)[0]  # Returns an integer

    # Convert integer prediction to crop name
    crop_name = {v: k for k, v in crop_dict.items()}.get(prediction, "Unknown Crop")
    
    return crop_name  #