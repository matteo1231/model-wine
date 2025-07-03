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

# Input widgets
with st.form("wine_input"):
    col1, col2 = st.columns(2)
    inputs = {}
    
    with col1:
        inputs['fixed acidity'] = st.slider('Fixed Acidity', 4.0, 16.0, 10.0)
        inputs['volatile acidity'] = st.slider('Volatile Acidity', 0.10, 1.60, 0.50)
        inputs['citric acid'] = st.slider('Citric Acid', 0.00, 1.00, 0.25)
        inputs['residual sugar'] = st.slider('Residual Sugar', 0.5, 15.5, 2.0)
        inputs['chlorides'] = st.slider('Chlorides', 0.01, 0.20, 0.08)
        
    with col2:
        inputs['free sulfur dioxide'] = st.slider('Free SO₂', 1, 70, 15)
        inputs['total sulfur dioxide'] = st.slider('Total SO₂', 5, 300, 100)
        inputs['density'] = st.slider('Density', 0.98, 1.04, 0.995)
        inputs['pH'] = st.slider('pH', 2.70, 4.00, 3.30)
        inputs['sulphates'] = st.slider('Sulphates', 0.30, 2.00, 0.60)
        inputs['alcohol'] = st.slider('Alcohol %', 8.0, 15.0, 10.5)
    
    submitted = st.form_submit_button("Predict Quality")

# Prediction logic
if submitted:
    # Debug: Show what inputs we have
    st.write("User inputs:", inputs)
    
    # Create DataFrame to ensure proper feature ordering
    input_df = pd.DataFrame([inputs])
    
    # Reorder columns to exactly match training data
    try:
        # Ensure we only keep features the model expects (excluding 'is_good' if present)
        model_features = [col for col in feature_columns if col != 'is_good']
        input_df = input_df[model_features]
        
        # Debug: Show final features being sent to model
        st.write("Features being sent to model:", input_df.columns.tolist())
        
        # Convert to numpy array
        input_array = input_df.values
        
        # Handle missing values
        input_imputed = imputer.transform(input_array)
        
        # Make prediction
        pred = model.predict(input_imputed)[0]
        proba = model.predict_proba(input_imputed)[0]
        
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
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Feature mismatch details:")
        st.write(f"Model expects: {model_features}")
        st.write(f"Received: {input_df.columns.tolist()}")
# Sidebar info
st.sidebar.header("About")
st.sidebar.info("""
**Classification Criteria**:  
- ✅ **Good**: Quality rating ≥ 7  
- ❌ **Not Good**: Quality rating < 7  
                
Model accuracy: 92%  
Precision (Good): 89%
""")