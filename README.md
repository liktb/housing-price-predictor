# Real Estate Valuation & Housing Price Predictor

A production-grade machine learning regression pipeline built to predict housing prices using Random Forest algorithms, optimized for hackathon submissions.

## Performance Metrics
- **$R^2$ Score:** 0.8830 (Explains ~88.3% of price variance)
- **Mean Absolute Error (MAE):** $94,516.88
- **Root Mean Squared Error (RMSE):** $119,975.33
- **Median Absolute Error:** $79,562.62

## Project Structure
- `train.py`: Cleans missing values, drops text constraints, executes train/test splits, trains the Random Forest Regressor, evaluates performance metrics, and saves the serialized model.
- `predict.py`: Loads the pre-trained model artifact (`housing_model.pkl`) to execute instant inference on new housing data.
- `housing_model.pkl`: The frozen, serialized Random Forest model artifact.
- `requirements.txt`: Locked Python environment dependencies.

## How to Run Locally

1. **Clone or download this repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
