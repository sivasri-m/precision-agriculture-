import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# File paths
MODEL_PATH = "pickle/fertilizer_model.pkl"
ENCODER_PATH = "pickle/fertilizer_encoder.pkl"

# Check if model exists
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    print("Loading saved fertilizer model...")
    model = pickle.load(open(MODEL_PATH, "rb"))
    encode_ferti = pickle.load(open(ENCODER_PATH, "rb"))
else:
    print("Training new fertilizer model...")
    # Load dataset
    data = pd.read_csv("data/Fertilizer_Prediction.csv")
    data.rename(columns={'Humidity ': 'Humidity', 'Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer'}, inplace=True)

    # Encode categorical variables
    encode_soil = LabelEncoder()
    data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)
    encode_crop = LabelEncoder()
    data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)
    encode_ferti = LabelEncoder()
    data.Fertilizer = encode_ferti.fit_transform(data.Fertilizer)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Fertilizer']), data.Fertilizer, test_size=0.2, random_state=1)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("pickle", exist_ok=True)
    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(encode_ferti, open(ENCODER_PATH, "wb"))
    print("Fertilizer model trained and saved!")

# Prediction function
def recommend_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorus):
    # Load saved models
    model = pickle.load(open("pickle/fertilizer_model.pkl", "rb"))
    ferti_encoder = pickle.load(open("pickle/fertilizer_encoder.pkl", "rb"))
    soil_encoder = pickle.load(open("pickle/soil_encoder.pkl", "rb"))
    crop_encoder = pickle.load(open("pickle/crop_encoder.pkl", "rb"))

    # Convert categorical inputs to encoded values
    soil_type = soil_encoder.transform([soil_type])[0]
    crop_type = crop_encoder.transform([crop_type])[0]

    # Create feature list
    features = [[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorus]]

    # Predict fertilizer
    prediction = model.predict(features)[0]
    recommended_fertilizer = ferti_encoder.inverse_transform([prediction])[0]

    return recommended_fertilizer
