
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging


logging.basicConfig(level=logging.INFO)
def preprocess_data(x, y):
  """
  1. train test split
  2. feature scaling/standardisation
  """

  logging.info("Starting preprocess, splitting data into train/test sets..")

  #test generalisation on unseen data to prevent model memorisation

  #x_train data the model learns from
  #x_test data the model is evaluated on
  #y_train correct answers for training data
  #y_test corrrect answers for testing data

  x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, #20 percent of data for testing
    random_state=42 #same split on every run reproducible results
  )

  logging.info("data split into training and test sets")
  logging.info("training feature shape: ")
  logging.info(x_train.shape)
  logging.info("testing feature shape: ")
  logging.info(x_test.shape)

  #feature scaling
  #standard scaler: makes data have mean 0, makes sd 1, normalise all the features
  scaler = StandardScaler()

  #fit -> calculate mean and std from training data
  #transform --> scale the data
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)
  logging.info("Feature scaling completed, sample scaled values: " + str(x_train_scaled[:2]))

  return x_train_scaled, x_test_scaled, y_train, y_test, scaler

#testing
# from load_dataset import load_data
# x, y, _, _ = load_data()
# preprocess_data(x, y)