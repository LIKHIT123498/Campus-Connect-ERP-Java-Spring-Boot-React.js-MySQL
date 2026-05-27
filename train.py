import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\House-price-prediction\data\housing.csv")

# Display first 5 rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

# Convert categorical columns into numerical
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LinearRegression()

# Train Model



model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

# R2 Score



r2 = r2_score(y_test, predictions)

print("R2 Score:", r2)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("Model Saved Successfully")
