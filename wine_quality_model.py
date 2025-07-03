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
df = pd.read_csv('winequality-red.csv')

# Convert quality to binary classification
df['is_good'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
df = df.drop('quality', axis=1)

# Handle missing values with MICE imputation
imputer = IterativeImputer(random_state=42)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Train-test split
X = df_imputed.drop('is_good', axis=1)
y = df_imputed['is_good']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
joblib.dump(X.columns, 'feature_columns.joblib')