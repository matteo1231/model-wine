# wine_quality_model.py
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load and preprocess data
df = pd.read_csv('winequality-red.csv', sep=';')  # MUST use semicolon delimiter

# Convert quality to binary classification
df['is_good'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Show class distribution
good_percent = df['is_good'].mean() * 100
print(f"Premium wines: {good_percent:.1f}% of dataset")

# Separate features and target BEFORE imputation
X = df.drop(['quality', 'is_good'], axis=1)  # Features only
y = df['is_good']  # Target

# Handle missing values with MICE imputation (features only)
imputer = IterativeImputer(random_state=42, max_iter=20)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline with scaling and classifier
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced_subsample',
        min_samples_leaf=3,
        max_depth=10,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
print("\nModel Evaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Not Good', 'Good']))

# Save artifacts
joblib.dump(model, 'wine_model.joblib')
joblib.dump(imputer, 'imputer.joblib')
joblib.dump(X.columns, 'feature_columns.joblib')

print("\nArtifacts saved:")
print(f"- wine_model.joblib (Model)")
print(f"- imputer.joblib (Imputer for {X.shape[1]} features)")
print(f"- feature_columns.joblib (Feature names)")