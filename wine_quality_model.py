# wine_quality_model.py
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess data
df = pd.read_csv('winequality-red.csv')  # Explicit delimiter

# Convert quality to binary classification
df['is_good'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Separate features and target BEFORE imputation
X = df.drop(['quality', 'is_good'], axis=1)  # Features only
y = df['is_good']  # Target

# Handle missing values with MICE imputation (features only)
imputer = IterativeImputer(random_state=42)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Create pipeline with scaling and classifier
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=150,
        class_weight='balanced',
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, 'wine_model.joblib')
joblib.dump(imputer, 'imputer.joblib')
joblib.dump(X.columns, 'feature_columns.joblib')  # Feature names only

print("Model training complete! Artifacts saved:")
print(f"- wine_model.joblib (Model)")
print(f"- imputer.joblib (Imputer for {X.shape[1]} features)")
print(f"- feature_columns.joblib (Feature names)")