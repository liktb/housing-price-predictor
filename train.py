import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

print("--- 1. LOADING HOUSING DATASET ---")
df = pd.read_csv('housing.csv')
df = df.dropna()

print("\n--- 2. PREPARING FEATURES AND TARGET ---")
target_column = 'Price'
X = df.drop(columns=[target_column, 'Address'])
y = df[target_column]
X = pd.get_dummies(X)

print(f"Total rows in dataset: {len(df)}")

print("\n--- 3. DOING THE TRAIN/TEST SPLIT ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)} | Testing samples (Unseen exam): {len(X_test)}")

print("\n--- 4. TRAINING THE REGRESSION MODEL ---")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete!")

print("\n--- 5. ADVANCED MODEL EVALUATION METRICS ---")
predictions = model.predict(X_test)

# Calculate multiple regression metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
med_ae = median_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n📁 Regression Evaluation Report:")
print(f"• R² Score (Fit Quality, max 1.0)     : {r2:.4f}")
print(f"• Mean Absolute Error (MAE)          : ${mae:,.2f}")
print(f"• Root Mean Squared Error (RMSE)     : ${rmse:,.2f}  (Penalizes large errors more)")
print(f"• Median Absolute Error              : ${med_ae:,.2f} (Robust to outliers)")

print("\n--- 6. FEATURE IMPORTANCE ANALYSIS (What drove the prices?) ---")
# Extract feature importances from the Random Forest model
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Most Important Factors in Predicting House Prices:")
print(feature_importances.head(5).to_string(index=False))
import joblib

# Save the trained model to disk
joblib.dump(model, 'housing_model.pkl')
print("\n--- 6. MODEL SAVED SUCCESSFULLY ---")
print("Saved model file: 'housing_model.pkl' created in your folder!")