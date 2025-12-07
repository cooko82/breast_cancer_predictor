

import streamlit as st
import numpy as np
from src.predict import load_model, predict

model = load_model()

st.title("Breast Cancer Prediction System")

inputs =[]
for i in range(10):
  inputs.append(st.number_input
  (f"Feature {i+1}", value=0.0))

if st.button("Predict"):
  result = predict(model, inputs)
  if result ==0:
    st.write("Malignant")
  else:
    st.write("Benign")