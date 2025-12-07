import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.predict import load_model, predict 

#Load Model and Feature Names ---

# Load the trained model
model = load_model()


FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 
    'mean fractal dimension', 'radius se', 'texture se', 'perimeter se', 'area se', 
    'smoothness se', 'compactness se', 'concavity se', 'concave points se', 
    'symmetry se', 'fractal dimension se', 'worst radius', 'worst texture', 
    'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 
    'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]


st.title("Breast Cancer Prediction System")
st.write("Welcome. This system utilizes a trained machine learning model to assist in classifying breast masses. Please input the 30 computed characteristics (mean, standard error, and 'worst' values) derived from the **fine-needle aspirate (FNA) histology report**. The model will analyse these geometric and textural features to predict if the growth is likely **Malignant (Cancerous)** or **Benign (Non-cancerous)**. The Feature Importance chart below explains which measurements most influence the model's decision.")

st.header("New Prediction Input")

# Use a two-column layout for cleaner input form
cols = st.columns(2)
inputs = []

# Loop through all 30 features and place them in columns
for i, feature_name in enumerate(FEATURE_NAMES):
    # Determine the column (0 or 1)
    col_index = i % 2 
    with cols[col_index]:
        # Use the actual feature name for the label for better clarity
        feature_value = st.number_input(f"{i+1}. {feature_name}", value=0.0)
        inputs.append(feature_value)

st.divider() # Visual separator

if st.button("Predict Diagnosis", type="primary"):
    # Ensure all 30 inputs were collected (safety check)
    if len(inputs) == 30:
        result = predict(model, inputs)
        
        # Display the prediction result
        if result == 0:
            st.error("🚨 Prediction: Malignant (Cancerous)")
        else:
            st.success("✅ Prediction: Benign (Non-cancerous)")
            
    else:
        st.error("Error: Please ensure all 30 features have been input.")