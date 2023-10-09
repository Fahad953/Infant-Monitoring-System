#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
def split_and_convert_bp_column(data):
    # Split the Blood_Pressure column into systolic and diastolic columns
    data[['Systolic_BP', 'Diastolic_BP']] = data['Blood_Pressure'].str.split('/', expand=True)

    # Convert the new columns to numerical values (floats)
    data['Systolic_BP'] = pd.to_numeric(data['Systolic_BP'], errors='coerce')
    data['Diastolic_BP'] = pd.to_numeric(data['Diastolic_BP'], errors='coerce')

    # Drop the original Blood_Pressure column
    data.drop(columns=['Blood_Pressure'], inplace=True)


# In[3]:


import pandas as pd

def data_clean(datasetTrain):
    irrelevant_columns = ['Family_Income', 'Parental_Education','Sleep_Duration_Hrs', 'ID']
    columns_to_convert = ['Fever', 'Cough', 'Runny_Nose', 'Skin_Rash', 'Vomiting', 'Diarrhea']
    datasetTrain = datasetTrain.drop(columns=irrelevant_columns)
    datasetTrain[columns_to_convert] = datasetTrain[columns_to_convert].astype(int)
    datasetTrain['Feeding_Method'] = datasetTrain['Feeding_Method'].replace({'Formula': 2, 'Breastfed': 1, 'Mixed': 0})
    datasetTrain['Immunization_Status'] = datasetTrain['Immunization_Status'].replace({'Complete': 1, 'Incomplete': 0})
    
    return datasetTrain


# In[ ]:




