# GFGKIIT-spikers-Hyperthon
HeartDisease_Detection_model

Heart Disease Prediction Model

video link: https://drive.google.com/file/d/1_8ds8sq6-k4sm4eBgt8nkPeb1wTVnHSz/view?usp=sharing

This project uses machine learning to predict the likelihood of heart disease based on medical data. It combines a Random Forest Classifier with visualizations and insights to provide actionable results, such as disease name, causes, and treatments.

Table of Contents

Overview

Dataset

Model

Setup Instructions

How to Use

Features

Visualizations

Results

Future Enhancements

Overview

This repository demonstrates a machine learning approach to predicting heart disease. It includes:

Preprocessing medical data.

Training a Random Forest Classifier.

Visualizing feature importance and data distribution.

Mapping predictions to detailed disease insights.

Dataset

The model uses the UCI Heart Disease Dataset. The dataset includes features such as:

Age, Sex, Chest Pain Type (cp)

Resting Blood Pressure (trestbps)

Cholesterol Level (chol)

Maximum Heart Rate Achieved (thalach)

ST Depression (oldpeak)

And more...

Data Source

The dataset is fetched from the UCI Machine Learning Repository.

Model

The model uses a Random Forest Classifier, a robust machine learning algorithm, to predict whether a patient has heart disease (1) or not (0).

Training Data Split: 80% Training, 20% Testing.

Evaluation Metrics: Accuracy, Precision, Recall, F1-score.

Setup Instructions

Prerequisites

Python 3.8+

Required Python libraries:

pandas

numpy

seaborn

matplotlib

scikit-learn

joblib

ucimlrepo

Installation

Clone the repository:

git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

Install dependencies:

pip install -r requirements.txt

Run the Jupyter Notebook or Python scripts provided.

How to Use

1. Train the Model

Run the Jupyter Notebook to train the Random Forest Classifier. The notebook includes:

Data preprocessing.

Training and evaluation.

Visualization of results.

2. Input Custom Data

Provide custom inputs in the following format:

custom_input = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
custom_input_df = pd.DataFrame(custom_input, columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
custom_prediction = model.predict(custom_input_df)[0]
print(get_disease_info(custom_prediction))

3. Use the Saved Model

To save and load the model:

import joblib
joblib.dump(model, "heart_disease_model.pkl")
loaded_model = joblib.load("heart_disease_model.pkl")

Features

Data Preprocessing: Handles missing values and splits the data into training and testing sets.

Disease Insights: Maps predictions to disease names, causes, and treatments.

Visualization: Feature importance, target distribution, and confusion matrix.

Model Persistence: Save and load trained models using joblib.

Visualizations

Feature Importance:

Bar chart showing the contribution of each feature to the model's predictions.

Target Distribution:

Count plot of the number of cases with and without heart disease.

Confusion Matrix:

Visual representation of true vs. predicted values.

Results

Accuracy: Achieved over 85% accuracy on the test set.

Insights:

Prediction of disease presence or absence with detailed explanations.

Example output for a custom input:

Prediction: 1

    Disease Name: Presence of Heart Disease
    Cause: Plaque buildup, high cholesterol.
    Cure/Treatment: Medications, surgery, and lifestyle changes.

Future Enhancements

Integrate advanced algorithms like XGBoost or LightGBM.

Extend the dataset to include more features and samples.

Deploy the model as a REST API or web app for real-time predictions.

Creating Future UI for it.


