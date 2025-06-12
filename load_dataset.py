from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
print(data.keys())
print("----")
print(data.DESCR[:1000])

X = data.data
y = data.target

#view the shape and the features of dataset
print(X.shape) #569 samples, 30 numeric feature
print(data.feature_names) #descriptions of the tumour
print(data.target_names) # this is what we are trying to predict either malignant and bengin (target names)
print("First 5 target values (y):", y[:5]) #so the firsst 5 are all malignant