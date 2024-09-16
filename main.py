# Load the dataset
dataset_path = 'diabetes.csv'
data = pd.read_csv(dataset_path)

# Separate features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Feature engineering: Replace 0 values with median
X['Insulin'] = np.where(X['Insulin'] == 0, np.nan, X['Insulin'])
X['SkinThickness'] = np.where(X['SkinThickness'] == 0, np.nan, X['SkinThickness'])
X.fillna(X.median(), inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply data augmentation (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# CatBoost Model (with previously obtained best hyperparameters)
catboost_params = {
    'iterations': 498,
    'learning_rate': 0.0154,
    'depth': 10,
    'l2_leaf_reg': 0.2355,
}

# Define and train the CatBoost model
catboost_model = CatBoostClassifier(**catboost_params)
catboost_model.fit(X_train, y_train)

# Make predictions and evaluate performance
y_pred_catboost = catboost_model.predict(X_test)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
auc_catboost = roc_auc_score(y_test, catboost_model.predict_proba(X_test)[:, 1])
report_catboost = classification_report(y_test, y_pred_catboost)

# Print results
print("CatBoost Model Performance:")
print(f"Accuracy: {accuracy_catboost}")
print(f"ROC AUC: {auc_catboost}")
print(report_catboost)
