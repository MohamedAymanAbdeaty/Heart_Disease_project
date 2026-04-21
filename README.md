# Predicting Heart Disease using Machine Learning

📌 Project Overview

This project applies machine learning algorithms to predict whether a patient has heart disease or not based on their medical attributes. The goal is to build a robust classification model that maximizes Recall (sensitivity) to ensure that no patient with actual heart disease is misdiagnosed as healthy.

📊 Data Description

The dataset contains medical records of patients. It includes the following features (attributes):

Feature

Description

Type / Values

age

Age of the patient in years

Continuous

sex

Gender of the patient

1 = Male, 0 = Female

cp

Chest pain type

0 = Typical angina



1 = Atypical angina



2 = Non-anginal pain



3 = Asymptomatic

trestbps

Resting blood pressure (in mm Hg)

Continuous

chol

Serum cholesterol in mg/dl

Continuous

fbs

Fasting blood sugar > 120 mg/dl

1 = True, 0 = False

restecg

Resting electrocardiographic results

0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy

thalach

Maximum heart rate achieved

Continuous

exang

Exercise-induced angina

1 = Yes, 0 = No

oldpeak

ST depression induced by exercise relative to rest

Continuous

slope

The slope of the peak exercise ST segment

0 = Upsloping, 1 = Flat, 2 = Downsloping

ca

Number of major vessels colored by fluoroscopy

0 to 3

thal

Thalassemia

1 = Normal, 2 = Fixed defect, 3 = Reversable defect

target

(Target Variable) Heart disease status

1 = Disease present, 0 = No disease

🛠️ Methodology & Technologies Used

Exploratory Data Analysis (EDA): Visualized relationships between features using Seaborn and Matplotlib ( correlation heatmaps, bar charts for chest pain vs. target).

Modeling: Evaluated baseline machine learning models, primarily focusing on Logistic Regression and Random Forest.

Hyperparameter Tuning: Utilized RandomizedSearchCV and GridSearchCV to find the optimal parameters for the Logistic Regression model ( tuning the C parameter and solver).

Evaluation: Assessed the model using standard classification metrics:

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)

