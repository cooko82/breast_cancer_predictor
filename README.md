# Breast Cancer Prediction System (Numeric Model)

## Overview

This project is a machine learning pipeline that predicts whether a breast tumor is **malignant** or **benign** based on numeric diagnostic measurements.

The model uses the **Breast Cancer Wisconsin Diagnostic Dataset**, which contains 569 patient samples and 30 real-valued features calculated from digitized images of fine needle aspirates (FNA) of breast masses.

This project demonstrates full end-to-end ML workflow, including:

- Dataset ingestion
- Data preprocessing
- Model training
- Model evaluation
- Model serialization
- API-style prediction
- Interactive UI using Streamlit

---

##UI
<img width="1470" height="712" alt="image" src="https://github.com/user-attachments/assets/12f72efd-f651-40bc-9ff9-3a09aced7d65" />
<img width="1470" height="712" alt="image" src="https://github.com/user-attachments/assets/d2d9fb11-4c52-4852-be22-19664f895c2c" />


## Project Architecture

breast-cancer-predictor/
├── models/
│ └── cancer_model.pkl
├── src/
│ ├── load_dataset.py
│ ├── preprocess.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── predict.py
├── ui/
│ └── app.py
├── run_pipeline.py
└── requirements.txt


---

## Dataset Details

- Source: `sklearn.datasets.load_breast_cancer()`
- Number of Samples: 569
- Number of Features: 30
- Target Classes:
  - `0 → malignant`
  - `1 → benign`

### Feature Examples
mean radius
mean texture
mean perimeter
worst area
worst concavity
...

---

## Model Architecture

- Algorithm: `Logistic Regression` (or your chosen classifier)
- Scaler: `StandardScaler`
- Train/Test Split: `80/20`
- Evaluation Metrics:
  - Accuracy
  - Confusion Matrix
  - Classification Report

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
### 2. Train model
python run_pipeline.py
### 3. Launch the UI
streamlit run ui/app.py


⚠️ This is NOT a medical device and must not be used for clinical decisions. lol

