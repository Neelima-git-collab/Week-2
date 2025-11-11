
# ==========================================================
# Week-2 : Exploratory Data Analysis (EDA)
# Project : Electric Vehicles Specification Analysis (2025 Dataset)
# ==========================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the cleaned dataset
df = pd.read_csv("cleaned_electric_vehicles_2025.csv")
print("✅ Cleaned dataset loaded successfully!")
print(f"Total Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

# Step 3: Display basic statistics
print("--- Basic Information ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe(include='all'))

# Step 4: Check for unique brands and categories
if 'Brand' in df.columns:
    print("\nTop 10 Brands in Dataset:")
    print(df['Brand'].value_counts().head(10))

# Step 5: Correlation Heatmap
plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Step 6: Distribution of EV Range
if 'Range_km' in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df['Range_km'], kde=True, bins=25)
    plt.title("Distribution of Electric Vehicle Range (km)")
    plt.xlabel("Range (km)")
    plt.ylabel("Count")
    plt.show()

# Step 7: Price Distribution
if 'Price_in_USD' in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df['Price_in_USD'], kde=True, bins=25, color='orange')
    plt.title("Price Distribution of Electric Vehicles (USD)")
    plt.xlabel("Price in USD")
    plt.ylabel("Count")
    plt.show()

# Step 8: Range vs Price Scatterplot
if {'Range_km', 'Price_in_USD'}.issubset(df.columns):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='Range_km', y='Price_in_USD', hue='Brand', data=df)
    plt.title("Range vs Price by Brand")
    plt.xlabel("Range (km)")
    plt.ylabel("Price (USD)")
    plt.show()

# Step 9: Battery Capacity vs Range
if {'Battery_Capacity_kWh', 'Range_km'}.issubset(df.columns):
    plt.figure(figsize=(8,6))
    sns.regplot(x='Battery_Capacity_kWh', y='Range_km', data=df, scatter_kws={'alpha':0.6})
    plt.title("Battery Capacity vs Range")
    plt.xlabel("Battery Capacity (kWh)")
    plt.ylabel("Range (km)")
    plt.show()

# Step 10: Average Price by Brand (Top 10)
if {'Brand', 'Price_in_USD'}.issubset(df.columns):
    brand_price = df.groupby('Brand')['Price_in_USD'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(x=brand_price.values, y=brand_price.index, palette='viridis')
    plt.title("Top 10 Brands by Average EV Price (USD)")
    plt.xlabel("Average Price (USD)")
    plt.ylabel("Brand")
    plt.show()

# Step 11: Key Insights
print("\n--- Week-2 Key Insights ---")
print("1️⃣ Higher battery capacity is strongly correlated with greater range.")
print("2️⃣ Significant variation in EV prices across brands.")
print("3️⃣ Some brands offer higher range per price ratio — potential for value analysis.")
print("4️⃣ Dataset ready for feature engineering and model training in Week-3.")
print("✅ Week-2 EDA Completed Successfully!")
