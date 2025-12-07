
import joblib
from sklearn.linear_model import LogisticRegression
import os
import logging

logging.basicConfig(level=logging.INFO)

def train_and_save_mdl(x_train, y_train):
  """
  training using logistic regression
  - best for binary classification
  """
  
  logging.info("starting model training")

  #create the model object
  model = LogisticRegression(max_iter=1000)

  #model.fit() learning for model analysing data patterns
  #learns the relationship between x and y
  model.fit(x_train, y_train)
  logging.info("model training complete")

  os.makedirs("models", exist_ok=True)

  #save trained model to disk
  joblib.dump(model, "models/c_pred_model.pkl")
  logging.info("Model saved to models folder")

  return model

# from load_dataset import load_data
# from preprocess import preprocess_data

# if __name__ == "__main__":
#     x, y, _, _ = load_data()
#     x_train, X_test, y_train, y_test, _ = preprocess_data(x, y)
#     train_and_save_mdl(x_train, y_train)