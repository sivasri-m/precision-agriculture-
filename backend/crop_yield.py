import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor

# File paths
MODEL_PATH = "pickle/yield_model.pkl"
PREPROCESSOR_PATH = "pickle/yield_preprocessor.pkl"

# Check if model exists
if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    print("Loading saved yield model...")
    dtr = pickle.load(open(MODEL_PATH, "rb"))
    preprocessor = pickle.load(open(PREPROCESSOR_PATH, "rb"))
else:
    print("Training new yield model...")
    # Load dataset
    df = pd.read_csv("data/yield_df.csv")
    df.drop(['Unnamed: 0'], axis=1, inplace=True, errors='ignore')
    df.drop_duplicates(inplace=True)
    df['average_rain_fall_mm_per_year'] = pd.to_numeric(df['average_rain_fall_mm_per_year'], errors='coerce')
    df.dropna(inplace=True)

    # Feature selection
    col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
    df = df[col]
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), [0, 1, 2, 3]),
            ('encode', OneHotEncoder(drop='first'), [4, 5])
        ]
    )
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Train model
    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, y_train)

    # Save model
    os.makedirs("pickle", exist_ok=True)
    pickle.dump(dtr, open(MODEL_PATH, "wb"))
    pickle.dump(preprocessor, open(PREPROCESSOR_PATH, "wb"))
    print("Yield model trained and saved!")

# Prediction function
def predict_yield(year, rainfall, pesticides, temp, area, item):
    features = [[year, rainfall, pesticides, temp, area, item]]  # Wrap inputs in a list
    transformed_features = preprocessor.transform(features)
    return dtr.predict(transformed_features)[0]
