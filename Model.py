#!/usr/bin/env python
# coding: utf-8

import sys
import joblib
import pandas as pd
from Function_Calling import split_and_convert_bp_column, data_clean

def main(argv):
    # if len(argv) != 20:
    #     print("Usage: python script.py <Age_Months> <Weight_Kg> <Height_Cm> <Oxygen_Saturation> <Pulse_Rate> <Temperature_C> <Fever> <Respiratory_Rate> <Cough> <Runny_Nose> <Skin_Rash> <Vomiting> <Diarrhea> <Blood_Pressure> <Sleep_Duration_Hrs> <Feeding_Method> <Immunization_Status> <Hygiene_Score> <Parental_Education> <Family_Income>")
    #     sys.exit(1)
    # Extract command-line arguments
    Age_Months, Weight_Kg, Height_Cm, Oxygen_Saturation, Pulse_Rate, Temperature_C, Fever, Respiratory_Rate, Cough, Runny_Nose, Skin_Rash, Vomiting, Diarrhea, Blood_Pressure, Sleep_Duration_Hrs, Feeding_Method, Immunization_Status, Hygiene_Score, Parental_Education, Family_Income = argv

    # Load the model from the file
    loaded_model = joblib.load('dt_model.joblib')

    # Create a DataFrame with the provided data
    data = {
        'ID':0,
        'Age_Months': [float(Age_Months)],
        'Weight_Kg': [float(Weight_Kg)],
        'Height_Cm': [float(Height_Cm)],
        'Oxygen_Saturation': [float(Oxygen_Saturation)],
        'Pulse_Rate': [float(Pulse_Rate)],
        'Temperature_C': [float(Temperature_C)],
        'Fever': [int(Fever)],
        'Respiratory_Rate': [int(Respiratory_Rate)],
        'Cough': [int(Cough)],
        'Runny_Nose': [int(Runny_Nose)],
        'Skin_Rash': [int(Skin_Rash)],
        'Vomiting': [int(Vomiting)],
        'Diarrhea': [int(Diarrhea)],
        'Blood_Pressure': [str(Blood_Pressure)],
        'Sleep_Duration_Hrs': [float(Sleep_Duration_Hrs)],
        'Feeding_Method': [str(Feeding_Method)],
        'Immunization_Status': [str(Immunization_Status)],
        'Hygiene_Score': [int(Hygiene_Score)],
        'Parental_Education': [str(Parental_Education)],
        'Family_Income': [float(Family_Income)],
    }

    df = pd.DataFrame(data)
    test_csv = 'data.csv'
 
# Write the DataFrame to a CSV file
    df.to_csv(test_csv, index=False) 
    df = pd.read_csv(test_csv)
    # # Split data set
    split_and_convert_bp_column(df)

    # # Data Cleaning
    X_testA = data_clean(df)


    # # Prediction using the loaded model
    predA = loaded_model.predict(X_testA)

    # # Print the results
    print(predA)
if __name__ == "__main__":
    main(sys.argv[1:])
