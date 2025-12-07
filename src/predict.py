"""
loads saved model to make predictions


notes
INFO:root:Loading model
INFO:root:input reshaped [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]] --> value of zeros means there is no variation and everything is normal for the features
Prediction result: [1] --> benging

"""

import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def load_model():
  logging.info("Loading model")
  return joblib.load("models/c_pred_model.pkl")

def predict(model, input_data):
  input_np = np.array(input_data).reshape(1, -1)
  logging.info("input reshaped "+str(input_np)) #t converts the single-sample feature list into a 2-dimensional NumPy array of shape > satisfying the input matrix requirement of scikit-learn predict()
  return model.predict(input_np)[0]

# if __name__ == "__main__":
#     model = load_model()
#     dummy_input = [0] * 30
#     prediction = predict(model, dummy_input)
#     print("Prediction result:", prediction)