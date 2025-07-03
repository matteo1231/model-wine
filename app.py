# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load artifacts
model = joblib.load('wine_model.joblib')
imputer = joblib.load('imputer.joblib')
feature_columns = joblib.load('feature_columns.joblib')

st.title("🍷 Premium Wine Quality Classifier")
st.subheader("Predict if wine meets premium standards (7+ rating)")

# Premium sample values for quick testing
PREMIUM_SAMPLE = {
    'fixed acidity': 11.5,
    'volatile acidity': 0.22,
    'citric acid': 0.65,
    'residual sugar': 1.8,
    'chlorides': 0.038,
    'free sulfur dioxide': 15,
    'total sulfur dioxide': 85,
    'density': 0.994,
    'pH': 3.10,
    'sulphates': 1.0,
    'alcohol': 13.5
}

# Input widgets
with st.form("wine_input"):
    col1, col2 = st.columns(2)
    inputs = {}
    
    with col1:
        inputs['fixed acidity'] = st.slider('Fixed Acidity', 4.0, 16.0, PREMIUM_SAMPLE['fixed acidity'])
        inputs['volatile acidity'] = st.slider('Volatile Acidity', 0.10, 1.60, PREMIUM_SAMPLE['volatile acidity'])
        inputs['citric acid'] = st.slider('Citric Acid', 0.00, 1.00, PREMIUM_SAMPLE['citric acid'])
        inputs['residual sugar'] = st.slider('Residual Sugar', 0.5, 15.5, PREMIUM_SAMPLE['residual sugar'])
        inputs['chlorides'] = st.slider('Chlorides', 0.01, 0.20, PREMIUM_SAMPLE['chlorides'])
        
    with col2:
        inputs['free sulfur dioxide'] = st.slider('Free SO₂', 1, 70, PREMIUM_SAMPLE['free sulfur dioxide'])
        inputs['total sulfur dioxide'] = st.slider('Total SO₂', 5, 300, PREMIUM_SAMPLE['total sulfur dioxide'])
        inputs['density'] = st.slider('Density', 0.98, 1.04, PREMIUM_SAMPLE['density'])
        inputs['pH'] = st.slider('pH', 2.70, 4.00, PREMIUM_SAMPLE['pH'])
        inputs['sulphates'] = st.slider('Sulphates', 0.30, 2.00, PREMIUM_SAMPLE['sulphates'])
        inputs['alcohol'] = st.slider('Alcohol %', 8.0, 15.0, PREMIUM_SAMPLE['alcohol'])
    
    submitted = st.form_submit_button("Predict Quality")

# Prediction logic
if submitted:
    try:
        # Create array with correct feature order
        input_array = np.array([[inputs[col] for col in feature_columns]])
        
        # Handle missing values
        input_imputed = imputer.transform(input_array)
        
        # Make prediction
        pred = model.predict(input_imputed)[0]
        proba = model.predict_proba(input_imputed)[0]
        
        # Debug: Show scaled features
        scaled = model.named_steps['scaler'].transform(input_imputed)
        st.write("Scaled Features:", scaled[0])
        
        # Display results
        st.subheader("Prediction Result")
        if pred == 1:
            st.success(f"✅ **Good Quality** (Confidence: {proba[1]:.1%})")
            st.markdown("This wine meets our premium standards!")
        else:
            st.error(f"❌ **Not Good Quality** (Confidence: {proba[0]:.1%})")
            st.markdown("Does not meet premium quality threshold")
        
        st.progress(proba[1])
        st.caption(f"Probability of being 'Good Quality': {proba[1]:.1%}")
        
        # Feature importance analysis
        if hasattr(model.named_steps['clf'], 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.named_steps['clf'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.subheader("Top Quality Indicators")
            st.bar_chart(importance.set_index('Feature').head(5))
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("""
**Classification Criteria**:  
- ✅ **Good**: Quality rating ≥ 7  
- ❌ **Not Good**: Quality rating < 7  

Premium wines typically have:  
- Volatile Acidity < 0.4  
- Alcohol > 12%  
- Sulphates > 0.6  
""")# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load artifacts
model = joblib.load('wine_model.joblib')
imputer = joblib.load('imputer.joblib')
feature_columns = joblib.load('feature_columns.joblib')

st.title("🍷 Premium Wine Quality Classifier")
st.subheader("Predict if wine meets premium standards (7+ rating)")

# Premium sample values for quick testing
PREMIUM_SAMPLE = {
    'fixed acidity': 11.5,
    'volatile acidity': 0.22,
    'citric acid': 0.65,
    'residual sugar': 1.8,
    'chlorides': 0.038,
    'free sulfur dioxide': 15,
    'total sulfur dioxide': 85,
    'density': 0.994,
    'pH': 3.10,
    'sulphates': 1.0,
    'alcohol': 13.5
}

# Input widgets
with st.form("wine_input"):
    col1, col2 = st.columns(2)
    inputs = {}
    
    with col1:
        inputs['fixed acidity'] = st.slider('Fixed Acidity', 4.0, 16.0, PREMIUM_SAMPLE['fixed acidity'])
        inputs['volatile acidity'] = st.slider('Volatile Acidity', 0.10, 1.60, PREMIUM_SAMPLE['volatile acidity'])
        inputs['citric acid'] = st.slider('Citric Acid', 0.00, 1.00, PREMIUM_SAMPLE['citric acid'])
        inputs['residual sugar'] = st.slider('Residual Sugar', 0.5, 15.5, PREMIUM_SAMPLE['residual sugar'])
        inputs['chlorides'] = st.slider('Chlorides', 0.01, 0.20, PREMIUM_SAMPLE['chlorides'])
        
    with col2:
        inputs['free sulfur dioxide'] = st.slider('Free SO₂', 1, 70, PREMIUM_SAMPLE['free sulfur dioxide'])
        inputs['total sulfur dioxide'] = st.slider('Total SO₂', 5, 300, PREMIUM_SAMPLE['total sulfur dioxide'])
        inputs['density'] = st.slider('Density', 0.98, 1.04, PREMIUM_SAMPLE['density'])
        inputs['pH'] = st.slider('pH', 2.70, 4.00, PREMIUM_SAMPLE['pH'])
        inputs['sulphates'] = st.slider('Sulphates', 0.30, 2.00, PREMIUM_SAMPLE['sulphates'])
        inputs['alcohol'] = st.slider('Alcohol %', 8.0, 15.0, PREMIUM_SAMPLE['alcohol'])
    
    submitted = st.form_submit_button("Predict Quality")

# Prediction logic
if submitted:
    try:
        # Create array with correct feature order
        input_array = np.array([[inputs[col] for col in feature_columns]])
        
        # Handle missing values
        input_imputed = imputer.transform(input_array)
        
        # Make prediction
        pred = model.predict(input_imputed)[0]
        proba = model.predict_proba(input_imputed)[0]
        
        # Debug: Show scaled features
        scaled = model.named_steps['scaler'].transform(input_imputed)
        st.write("Scaled Features:", scaled[0])
        
        # Display results
        st.subheader("Prediction Result")
        if pred == 1:
            st.success(f"✅ **Good Quality** (Confidence: {proba[1]:.1%})")
            st.markdown("This wine meets our premium standards!")
        else:
            st.error(f"❌ **Not Good Quality** (Confidence: {proba[0]:.1%})")
            st.markdown("Does not meet premium quality threshold")
        
        st.progress(proba[1])
        st.caption(f"Probability of being 'Good Quality': {proba[1]:.1%}")
        
        # Feature importance analysis
        if hasattr(model.named_steps['clf'], 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.named_steps['clf'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.subheader("Top Quality Indicators")
            st.bar_chart(importance.set_index('Feature').head(5))
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("""
**Classification Criteria**:  
- ✅ **Good**: Quality rating ≥ 7  
- ❌ **Not Good**: Quality rating < 7  

Premium wines typically have:  
- Volatile Acidity < 0.4  
- Alcohol > 12%  
- Sulphates > 0.6  
""")