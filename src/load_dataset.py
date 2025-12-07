"""
-loading the dataset from sckitlearn
- returning clean feature and label arrays
- separative from data and ml logic
"""

from sklearn.datasets import load_breast_cancer

def load_data():
    """loads ds from sklearn
    output
    x (ndarray): feature matrix (shape: [samples, features])
    y (ndarray): target vector (0 = bad. 1=benign)
    feature names (list): names of input features
    target names (list): output class names
    """
    data = load_breast_cancer()
    print(data.keys())
    print("----")
    print(data.DESCR[:1000])

    x = data.data
    print("X: --") #features
    print(x)
    y = data.target
    print("Y ---") #bad or good
    print(y)
    feature_names = data.feature_names
    target_names = data.target_names

    #view the shape and the features of dataset
    print(f"X_shape: {x.shape}") #569 samples, 30 numeric feature
    print(f"Feature names: {data.feature_names}") #descriptions of the tumour
    print(f"target names {data.target_names}") # this is what we are trying to predict either malignant and bengin (target names)
    print("First 5 target values (y):", y[:5]) #so the firsst 5 are all malignant
    return x, y, feature_names, target_names

# load_data()