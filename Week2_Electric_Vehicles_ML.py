
# ==========================================================
# Week-2 : Machine Learning Model Building
# Project : Electric Vehicles Specification Analysis (2025 Dataset)
# ==========================================================

# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# Step 2: Load the cleaned dataset
df = pd.read_csv("cleaned_electric_vehicles_2025.csv")
print("✅ Dataset loaded successfully!")
print(f"Total Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

# Step 3: Basic preprocessing
df = df.dropna(subset=['Price_in_USD', 'Range_km'])
df = df.fillna(df.mean(numeric_only=True))

# Encode categorical features
label_enc = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_enc.fit_transform(df[col].astype(str))

print("✅ Data preprocessing completed.\n")

# ==========================================================
# MODEL 1: Predict Price_in_USD
# ==========================================================

if 'Price_in_USD' in df.columns:
    print("🚗 Training Model 1: Predicting EV Price (USD)")
    X_price = df.drop(['Price_in_USD'], axis=1)
    y_price = df['Price_in_USD']

    X_train, X_test, y_train, y_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

    # Linear Regression
    lin_reg_price = LinearRegression()
    lin_reg_price.fit(X_train, y_train)
    y_pred_lr = lin_reg_price.predict(X_test)

    # Random Forest Regressor
    rf_price = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_price.fit(X_train, y_train)
    y_pred_rf = rf_price.predict(X_test)

    # Evaluate models
    def evaluate_model(y_true, y_pred, name):
        print(f"\n📊 {name} Performance:")
        print(f"R2 Score: {r2_score(y_true, y_pred):.3f}")
        print(f"MAE: {mean_absolute_error(y_true, y_pred):.3f}")
        print(f"MSE: {mean_squared_error(y_true, y_pred):.3f}")

    evaluate_model(y_test, y_pred_lr, "Linear Regression (Price)")
    evaluate_model(y_test, y_pred_rf, "Random Forest (Price)")

    # Save best model
    with open("price_model.pkl", "wb") as f:
        pickle.dump(rf_price, f)
    print("✅ Price prediction model saved as price_model.pkl\n")

# ==========================================================
# MODEL 2: Predict Range_km
# ==========================================================

if 'Range_km' in df.columns:
    print("🔋 Training Model 2: Predicting EV Range (km)")
    X_range = df.drop(['Range_km'], axis=1)
    y_range = df['Range_km']

    X_train, X_test, y_train, y_test = train_test_split(X_range, y_range, test_size=0.2, random_state=42)

    # Linear Regression
    lin_reg_range = LinearRegression()
    lin_reg_range.fit(X_train, y_train)
    y_pred_lr = lin_reg_range.predict(X_test)

    # Random Forest Regressor
    rf_range = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_range.fit(X_train, y_train)
    y_pred_rf = rf_range.predict(X_test)

    # Evaluate models
    evaluate_model(y_test, y_pred_lr, "Linear Regression (Range)")
    evaluate_model(y_test, y_pred_rf, "Random Forest (Range)")

    # Save best model
    with open("range_model.pkl", "wb") as f:
        pickle.dump(rf_range, f)
    print("✅ Range prediction model saved as range_model.pkl\n")

print("🎯 Week-2 Machine Learning Task Completed Successfully!")
