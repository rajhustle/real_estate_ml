import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np

# Load data with low_memory=False to reduce dtype warnings
df = pd.read_csv(r'D:\raj\AIML\real_estate\properties.csv', low_memory=False)

# Select only columns needed for features and target
feature_cols = ['City', 'Type of Property', 'Units Available', 'Carpet Area']  # Adjust if needed
target_col = 'Price'

# Basic data quality checks
print("Initial shape:", df.shape)
print("Missing values per column:\n", df[feature_cols + [target_col]].isnull().sum())

# Drop rows where target or key features are missing
df = df.dropna(subset=feature_cols + [target_col])
print("Shape after dropping missing values:", df.shape)

# Fix data types if needed, for example convert Units Available to numeric
df['Units Available'] = pd.to_numeric(df['Units Available'], errors='coerce')
df['Carpet Area'] = pd.to_numeric(df['Carpet Area'], errors='coerce')

# After conversion, drop rows where conversion failed (NaNs introduced)
df = df.dropna(subset=['Units Available', 'Carpet Area'])
print("Shape after dtype fix:", df.shape)

# Optional: Remove outliers in Price or covered area (example using IQR)
Q1 = df[target_col].quantile(0.25)
Q3 = df[target_col].quantile(0.75)
IQR = Q3 - Q1
df = df[(df[target_col] >= Q1 - 1.5 * IQR) & (df[target_col] <= Q3 + 1.5 * IQR)]
print("Shape after removing price outliers:", df.shape)

# Prepare features X and target y
X = df[feature_cols]
y = df[target_col]

# One-hot encode categorical features
categorical_features = ['City', 'Type of Property']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # numeric features pass through
)

# Transform features
X_processed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessor
joblib.dump(model, 'rf_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')

print("Preprocessing done and model saved!")
