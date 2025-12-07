from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
def evaluate_model(model, x_test, y_test):
  """evalutaes how well the model performs on unseen data"""
  logging.info("starting eval")

  #generate predictions from trained model
  y_pred = model.predict(x_test)

  logging.info("Accuracy score: ")
  logging.info(accuracy_score(y_test, y_pred))

  logging.info("Classifcation report (precision /recall / f1)")
  logging.info(classification_report(y_test, y_pred))

  logging.info("Confusion matrix")
  logging.info(confusion_matrix(y_test, y_pred))

