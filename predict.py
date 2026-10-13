# ==========================================
# MODEL INFERENCE / PREDICTION SCRIPT
# ==========================================

import pandas as pd
import joblib

print("--- 1. LOADING SAVED MODEL ---")
# Load the model we saved earlier in milliseconds (no retraining needed!)
model = joblib.load('housing_model.pkl')
print("Model loaded successfully!")

print("\n--- 2. CREATING NEW HOUSE DATA TO PREDICT ---")
# Imagine these are brand new houses that just hit the market. 
# We want our AI to predict their prices!
new_houses = pd.DataFrame({
    'Avg. Area Income': [75000.0, 62000.0, 95000.0],
    'Avg. Area House Age': [6.5, 5.0, 7.2],
    'Avg. Area Number of Rooms': [7.0, 6.0, 8.1],
    'Avg. Area Number of Bedrooms': [4.0, 3.0, 4.0],
    'Area Population': [35000.0, 28000.0, 42000.0]
})

print(new_houses)

print("\n--- 3. GENERATING PREDICTIONS ---")
# Use the loaded model to predict prices for these new houses
predicted_prices = model.predict(new_houses)

print("\n--- 4. FINAL PREDICTED RESULTS ---")
for i, price in enumerate(predicted_prices):
    print(f"House {i+1} Estimated Price: ${price:,.2f}")