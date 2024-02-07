# # Infant Monitory System
# 
# The Infant Monitoring System project powered by Machine Learning is a groundbreaking endeavor that combines cutting-edge technology with the utmost concern for infant safety and well-being. This project aims to create a sophisticated and intelligent system that assists caregivers in monitoring and ensuring the health and safety of infants in real-time.
# 
# Using state-of-the-art machine learning algorithms, this system is designed to process and analyze various data inputs strategically placed around the infant's environment. The machine learning models integrated into the system can learn and understand normal patterns of infant behavior and environmental conditions.

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from Function_Calling import split_and_convert_bp_column,data_clean

import warnings
warnings.filterwarnings('ignore')


# ### Data Preprocessing
# #### Data Collection
# In[2]
# Load the dataset
filename = "train.csv"
dataset = pd.read_csv(filename)
dataset.head()
dataset.dtypes
# #### Data Preprocessing
split_and_convert_bp_column(dataset)
# #### Data Cleaning
# 1. Handling Missing Values
dataset.dropna()  # Remove rows with any missing values
dataset.fillna(0)  # Replace missing values with a specified value
dataset.interpolate()  # Interpolate missing values based on neighboring values
# 2. Handling Duplicates
dataset.drop_duplicates()  # Remove duplicate rows
# 3. Addressing Irrelevant Data
irrelevant_columns = ['Family_Income', 'Parental_Education', 'Sleep_Duration_Hrs', 'ID']
dataset = dataset.drop(columns=irrelevant_columns)
# 4. Boolean Encoding
columns_to_convert = ['Fever', 'Cough', 'Runny_Nose', 'Skin_Rash', 'Vomiting', 'Diarrhea']
dataset[columns_to_convert] = dataset[columns_to_convert].astype(int)
dataset['Feeding_Method'] = dataset['Feeding_Method'].replace({'Formula': 2, 'Breastfed': 1, 'Mixed': 0})
dataset['Immunization_Status'] = dataset['Immunization_Status'].replace({'Complete': 1, 'Incomplete': 0})
dataset['Infant_Status'] = dataset['Infant_Status'].replace({'Healthy': 0, 'Sick': 1})
dataset.head()

# Summary Statistics
summary_stats = dataset.describe()
print(summary_stats)

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Age_Months'], bins=20, kde=True)
plt.title('Distribution of Age in Months')
plt.xlabel('Age (Months)')
plt.ylabel('Frequency')
plt.show()
# plotting with target feature
sns.countplot(data=dataset, x='Infant_Status', palette='Set2')
plt.title('Count of Infant Health Status')
plt.xlabel('Infant Status')
plt.ylabel('Count')
plt.show()
# Split the dataset into feature data (X) and label data (y)
X = dataset.drop(columns='Infant_Status')
y = dataset['Infant_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
# X_train=> This variable will contain the feature data for training the model.
# X_test=>  This variable will contain the feature data for testing the model.
# y_train: This variable will contain the target data for training.
# y_test: This variable will contain the target data for testing.
# ### Model Selection
# A random seed is an initial value used in various algorithms, such as random number generators, that helps generate a sequence of pseudo-random numbers.
# Train and Predict using Different Models
# Create a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)   # n_estimators: The number of decision trees in the forest. random_state: A random seed for reproducibility.
model.fit(X_train, y_train)
rf_pred = model.predict(X_test)
# Logistic Regression
logreg_model = LogisticRegression(random_state=42)  #  random_state: A random seed for reproducibility.
logreg_model.fit(X_train, y_train)
logreg_pred = logreg_model.predict(X_test)
# Support Vector Machine (SVM)
svm_model = SVC(random_state=42)  #  random_state: A random seed for reproducibility.
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)    #  random_state: A random seed for reproducibility.   # n_estimators: The number of decision trees in the forest. 
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
# Neural Network (Multi-layer Perceptron)
nn_model = MLPClassifier(random_state=42)  #  random_state: A random seed for reproducibility.
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)
# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)  #  random_state: A random seed for reproducibility.
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
# AdaBoost
adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=42)    #  random_state: A random seed for reproducibility.   # n_estimators: The number of decision trees in the forest. 
adaboost_model.fit(X_train, y_train)
adaboost_pred = adaboost_model.predict(X_test)
# Print predictions for each model
print("Logistic Regression Predictions:", logreg_pred)
print("SVM Predictions:", svm_pred)
print("Gradient Boosting Predictions:", gb_pred)
print("Naive Bayes Predictions:", nb_pred)
print("Neural Network Predictions:", nn_pred)
print("Decision Tree Predictions:", dt_pred)
print("AdaBoost Predictions:", adaboost_pred)
print("Random Forest Predictions:", rf_pred)
# The accuracy_score function from scikit-learn (sklearn) is a metric used to evaluate the accuracy of a classification model. It compares the predicted class labels generated by a machine learning model with the actual true class labels to determine how many predictions were correct. It's a common and straightforward metric for evaluating the overall performance of classification models.
# Calculate Accuracy for Each Model
logreg_accuracy = accuracy_score(y_test, logreg_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
nb_accuracy = accuracy_score(y_test, nb_pred)
nn_accuracy = accuracy_score(y_test, nn_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
adaboost_accuracy = accuracy_score(y_test, adaboost_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
# Print Accuracy, Confusion matrix and Classification report for Each Model
print(f"Logistic Regression Accuracy: {logreg_accuracy * 100:.2f}%")
print(f'Confusion matrix :\n {confusion_matrix(logreg_pred, y_test)}')

print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
print(f'Confusion matrix :\n {confusion_matrix(svm_pred, y_test)}')

print(f"Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%")
print(f'Confusion matrix :\n {confusion_matrix(gb_pred, y_test)}')

print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
print(f'Confusion matrix :\n {confusion_matrix(nb_pred, y_test)}')

print(f"Neural Network Accuracy: {nn_accuracy * 100:.2f}%")
print(f'Confusion matrix :\n {confusion_matrix(nn_pred, y_test)}')

print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print(f'Confusion matrix :\n {confusion_matrix(dt_pred, y_test)}')

print(f"AdaBoost Accuracy: {adaboost_accuracy * 100:.2f}%")
print(f'Confusion matrix :\n {confusion_matrix(adaboost_pred, y_test)}')

print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f'Confusion matrix :\n {confusion_matrix(rf_pred, y_test)}')
# Calculate ROC curve and AUC
y_prob = dt_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
# Define hyperparameter grids for different models
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'max_iter': [100, 200, 300]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1]
}

param_grid_nn = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'alpha': [0.0001, 0.001, 0.01]
}

param_grid_dt = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1]
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create GridSearchCV instances for different models
grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5)
grid_search_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5)
grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=5)
grid_search_nn = GridSearchCV(MLPClassifier(random_state=42), param_grid_nn, cv=5)
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_search_adaboost = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid_adaboost, cv=5)
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)

# List of models and their corresponding grid search instances
models = [
    ("Logistic Regression", grid_search_lr),
    ("SVM", grid_search_svm),
    ("Gradient Boosting", grid_search_gb),
    ("Neural Network", grid_search_nn),
    ("Decision Tree", grid_search_dt),
    ("AdaBoost", grid_search_adaboost),
    ("Random Forest", grid_search_rf)
]

for model_name, grid_search in models:
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2%}")
    print(f"Best Parameters: {best_params}")
    print("=" * 40)
# Initialize a list to store results for each model
results = []
# Iterate over each model and its corresponding grid search instance
for model_name, grid_search in models:
    if model_name == "Naive Bayes":
        best_model = grid_search
        best_model.fit(X_train, y_train)  # Fit Naive Bayes model
        y_pred = best_model.predict(X_test)
    else:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    
    # Classification report
    classification_rep = classification_report(y_test, y_pred)
    
    # Store results in the list
    results.append({
        "Model": model_name,
        "Best Model": best_model,
        "Accuracy": accuracy,
        "Confusion Matrix": confusion
    })

# Print the results for each model
for result in results:
    print(f"Model: {result['Model']}")
    print(f"Accuracy: {result['Accuracy']:.2%}")
    print("Confusion Matrix:")
    print(result['Confusion Matrix'])
    print("Classification Report:")
    print("=" * 40)
# #### Model Saving
# Save the model to a file
filename = 'Model.joblib'
joblib.dump(grid_search_dt, filename)
joblib.dump(dt_model, 'dt_model2.joblib')
# #### Model Calling 
# Load the model from the file
loaded_model = joblib.load('Model.joblib')
# Load Test Data set
filename2 = "convert.csv"
datasetTrain = pd.read_csv(filename2)
split_and_convert_bp_column(datasetTrain)
X_testA = data_clean(datasetTrain)
predA = dt_model.predict(X_testA)
# Print the results
print("Loaded Model Prediction:", predA)
# ### Results