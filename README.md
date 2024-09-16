# Pima Indians Diabetes Classification with CatBoost

This project aims to predict whether a patient has diabetes or not based on the Pima Indians Diabetes dataset. The model used in this project is the **CatBoost Classifier**, a gradient boosting algorithm optimized for categorical data.

## Dataset

The dataset used for this project is the **Pima Indians Diabetes Database**, originally provided by the National Institute of Diabetes and Digestive and Kidney Diseases.

- **Dataset Source**: [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

### Dataset Description

- **Features**:
  - `Pregnancies`: Number of pregnancies
  - `Glucose`: Plasma glucose concentration
  - `BloodPressure`: Diastolic blood pressure (mm Hg)
  - `SkinThickness`: Triceps skin fold thickness (mm)
  - `Insulin`: 2-Hour serum insulin (mu U/ml)
  - `BMI`: Body mass index (weight in kg/(height in m)^2)
  - `DiabetesPedigreeFunction`: Diabetes pedigree function
  - `Age`: Age in years
  
- **Target Variable**:
  - `Outcome`: 1 if the patient has diabetes, 0 otherwise

## Model and Approach

The model used is **CatBoostClassifier**, an efficient implementation of gradient boosting that handles categorical features natively and provides superior performance with relatively less parameter tuning.

### Key Steps in the Pipeline

1. **Data Preprocessing**:
    - Missing or zero values for `Insulin` and `SkinThickness` were replaced by the median of the respective columns.
    - Feature scaling was applied using `StandardScaler`.
   
2. **SMOTE** (Synthetic Minority Over-sampling Technique):
    - SMOTE was used to address class imbalance by creating synthetic instances of the minority class (Outcome = 1).

3. **Model Training**:
    - The CatBoost model was trained using a set of optimized hyperparameters obtained via experimentation and hyperparameter tuning.

4. **Model Evaluation**:
    - Accuracy and ROC AUC score were used to evaluate the model’s performance.

### Optimized Hyperparameters for CatBoost:
- `iterations`: 498
- `learning_rate`: 0.0154
- `depth`: 10
- `l2_leaf_reg`: 0.2355

## Results

- **Accuracy**: 0.84
- **ROC AUC**: 0.892
- **Classification Report**:
          precision    recall  f1-score   support

       0              0.89      0.77      0.83       99
       1              0.80      0.91      0.85       101
      accuracy                            0.84       200
      macro avg       0.85      0.84      0.84       200 
      weighted avg    0.85      0.84      0.84       200



## Code

For the implementation details, please refer to the `main.py` file in this repository.
