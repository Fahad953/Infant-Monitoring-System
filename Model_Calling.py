#!/usr/bin/env python
# coding: utf-8

# In[3]:


import joblib
import pandas as pd
from Function_Calling import split_and_convert_bp_column,data_clean

# Load the model from the file
loaded_model = joblib.load('dt_model.joblib')

# Load Test Data set
filename2 = "convert.csv"
datasetTrain = pd.read_csv(filename2)
print(datasetTrain,"DATA before")
# Split data set
split_and_convert_bp_column(datasetTrain)

print(datasetTrain,"DATA")
# Data Cleaning
X_testA = data_clean(datasetTrain)

# Prediction using the loaded model
predA = loaded_model.predict(X_testA)

# Print the results
print("Loaded Model Prediction:", predA)

