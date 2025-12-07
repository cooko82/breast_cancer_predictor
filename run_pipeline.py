
from src.load_dataset import load_data
from src.preprocess import preprocess_data
from src.train_model import train_and_save_mdl
from src.evaluate_model import evaluate_model

import logging

logging.basicConfig(level=logging.INFO)

def main():
  logging.info("running full pipeline")

  logging.info("load dataset")
  x, y, _, _ = load_data()
  logging.info("preprocess data")
  x_train, x_test, y_train, y_test, _ = preprocess_data(x, y)
  logging.info("traiing model")
  model = train_and_save_mdl(x_train, y_train)
  logging.info("evaluating model")
  evaluate_model(model, x_test, y_test)
  logging.info("completed successfully")

# if __name__=="__main___":
main()