import pandas as pd
import random

# Define normal and abnormal ranges for each parameter
normal_ranges = {
    'Age_Months': (0, 24),
    'Weight_Kg': (2.5, 10),
    'Height_Cm': (45, 75),
    'Oxygen_Saturation': (95, 100),
    'Pulse_Rate': (100, 160),
    'Temperature_C': (36.5, 37.5),
    'Fever': (0, 1),
    'Respiratory_Rate': (30, 60),
    'Cough': (0, 1),
    'Runny_Nose': (0, 1),
    'Skin_Rash': (0, 1),
    'Vomiting': (0, 1),
    'Diarrhea': (0, 1),
    'Blood_Pressure': ('90/60', '120/80'),  # Represented as systolic/diastolic
    'Sleep_Duration_Hrs': (10, 18),
    'Feeding_Method': ('Formula', 'Breastfed'),  # 'Formula' or 'Breastfed'
    'Immunization_Status': ('Incomplete', 'Complete'),  # Represented as Complete or Incomplete
    'Hygiene_Score': (0, 10)
}

column_list = ['ID'] + list(normal_ranges.keys()) + ['Infant_Status']

# Load the existing CSV file if it exists
try:
    df = pd.read_csv('train.csv')
    next_id = len(df) + 1
except FileNotFoundError:
    # Create an empty DataFrame if the file doesn't exist
    df = pd.DataFrame(columns=column_list)
    next_id = 1

# Generate 1000 random records with abnormalities and append to the existing DataFrame
for i in range(next_id, next_id + 1000):
    record = {}
    slight_abnormality_count = 0
    severe_abnormality_count = 0

    for param, val_range in normal_ranges.items():
        min_val, max_val = val_range

        if param == 'Blood_Pressure':
            systolic = random.uniform(int(min_val.split('/')[0]), int(max_val.split('/')[0]))
            diastolic = random.uniform(int(min_val.split('/')[1]), int(max_val.split('/')[1]))
            record[param] = f"{systolic:.1f}/{diastolic:.1f}"
            
            if systolic < float(min_val.split('/')[0]) * 0.9 or systolic > float(max_val.split('/')[0]) * 1.1 or \
               diastolic < float(min_val.split('/')[1]) * 0.9 or diastolic > float(max_val.split('/')[1]) * 1.1:
                severe_abnormality_count += 1
            elif systolic < float(min_val.split('/')[0]) or systolic > float(max_val.split('/')[0]) or \
               diastolic < float(min_val.split('/')[1]) or diastolic > float(max_val.split('/')[1]):
                slight_abnormality_count += 1
        elif param == 'Age_Months':
            record[param] = random.randint(int(min_val), int(max_val))
        elif param in ['Weight_Kg', 'Height_Cm', 'Temperature_C']:
            val = round(random.uniform(min_val, max_val), 1)
            record[param] = val
            
            if val < min_val * 0.9 or val > max_val * 1.1:
                severe_abnormality_count += 1
            elif val < min_val or val > max_val:
                slight_abnormality_count += 1
        elif param == 'Immunization_Status':
            record[param] = random.choice(['Complete', 'Incomplete'])
        elif param == 'Feeding_Method':
            record[param] = random.choice(['Formula', 'Breastfed'])
        else:
            record[param] = random.randint(int(min_val), int(max_val))

        if param in ['Fever', 'Cough', 'Diarrhea', 'Vomiting', 'Runny_Nose', 'Skin_Rash']:
            if param == 'Fever':
                if record[param] == 1:
                    # When fever is present, ensure temperature is out of normal range
                    record['Temperature_C'] = round(random.uniform(38.5, 50), 1)
                    severe_abnormality_count += 1
            else:
                if record[param] == 1:
                    if param == 'Fever':
                        severe_abnormality_count += 1
                    else: 
                        slight_abnormality_count += 1

    # Determine Infant_Status (0 for healthy, 1 for sick)
    if severe_abnormality_count >= 1 or slight_abnormality_count >= 3:
        record['Infant_Status'] = 'Sick'
    else:
        record['Infant_Status'] = 'Healthy'

    # Convert 1 to True and 0 to False for specified columns
    for param in ['Fever', 'Cough', 'Diarrhea', 'Vomiting', 'Runny_Nose', 'Skin_Rash']:
        if param in record and record[param] == 1:
            record[param] = True
        else:
            record[param] = False

    # Add an auto-incremented ID column
    record['ID'] = i

    df = df.append(record, ignore_index=True)

# # Generate 500 rows with no severe abnormalities or with slight abnormalities between 1 and 2
# for i in range(next_id + 1000, next_id + 1500):
#     record = {}
    
#     for param, val_range in normal_ranges.items():
#         min_val, max_val = val_range
        
#         if param == 'Blood_Pressure':
#             systolic = random.uniform(int(min_val.split('/')[0]), int(max_val.split('/')[0]))
#             diastolic = random.uniform(int(min_val.split('/')[1]), int(max_val.split('/')[1]))
#             record[param] = f"{systolic:.1f}/{diastolic:.1f}"
#         elif param == 'Age_Months':
#             record[param] = random.randint(int(min_val), int(max_val))
#         elif param in ['Weight_Kg', 'Height_Cm', 'Temperature_C']:
#             val = round(random.uniform(min_val, max_val), 1)
#             record[param] = val
#         elif param == 'Immunization_Status':
#             record[param] = random.choice(['Complete', 'Incomplete'])
#         elif param == 'Feeding_Method':
#             record[param] = random.choice(['Formula', 'Breastfed'])
#         else:
#             record[param] = random.randint(int(min_val), int(max_val))
    
#     # Ensure there are no severe abnormalities or slight abnormalities between 1 and 2
#     slight_abnormality_count = 0
#     severe_abnormality_count = 0
    
#     if random.random() < 0.5:
#         # Generate a slight abnormality
#         slight_abnormality_count += 1
#         param = random.choice(list(normal_ranges.keys()))
#         if param == 'Blood_Pressure':
#             systolic = random.uniform(int(normal_ranges[param][0].split('/')[0]), int(normal_ranges[param][1].split('/')[0]))
#             diastolic = random.uniform(int(normal_ranges[param][0].split('/')[1]), int(normal_ranges[param][1].split('/')[1]))
#             record[param] = f"{systolic:.1f}/{diastolic:.1f}"
#         elif param == 'Age_Months':
#             record[param] = random.randint(int(normal_ranges[param][0]), int(normal_ranges[param][1]))
#         elif param in ['Weight_Kg', 'Height_Cm', 'Temperature_C']:
#             val = round(random.uniform(normal_ranges[param][0], normal_ranges[param][1]), 1)
#             record[param] = val
#         elif param == 'Immunization_Status':
#             record[param] = random.choice(['Complete', 'Incomplete'])
#         elif param == 'Feeding_Method':
#             record[param] = random.choice(['Formula', 'Breastfed'])
    
#     # Ensure 'Fever' is always set to False
#     record['Fever'] = False
    
#     # Convert 1 to True and 0 to False for specified columns
#     for param in ['Cough', 'Diarrhea', 'Vomiting', 'Runny_Nose', 'Skin_Rash']:
#         record[param] = bool(random.randint(0, 1))
    
#     # Determine Infant_Status (0 for healthy, 1 for sick)
#     slight_abnormality_count = 0
#     severe_abnormality_count = 0
#     for param, val_range in normal_ranges.items():
#         min_val, max_val = val_range

#         if param == 'Blood_Pressure':
#             systolic, diastolic = map(float, record[param].split('/'))
#             if systolic < float(min_val.split('/')[0]) or systolic > float(max_val.split('/')[0]) or \
#                diastolic < float(min_val.split('/')[1]) or diastolic > float(max_val.split('/')[1]):
#                 slight_abnormality_count += 1
#         elif param == 'Age_Months':
#             if record[param] < min_val or record[param] > max_val:
#                 slight_abnormality_count += 1
#         elif param in ['Weight_Kg', 'Height_Cm', 'Temperature_C']:
#             val = record[param]
#             if val < min_val or val > max_val:
#                 slight_abnormality_count += 1
#         elif param == 'Fever':
#             if record[param] == 1:
#                 slight_abnormality_count += 1
#         else:
#             if record[param] < min_val or record[param] > max_val:
#                 slight_abnormality_count += 1

#     # Determine Infant_Status (0 for healthy, 1 for sick)
#     if severe_abnormality_count >= 1 or slight_abnormality_count >= 3:
#         record['Infant_Status'] = 'Sick'
#     else:
#         record['Infant_Status'] = 'Healthy'
    
#     # Add an auto-incremented ID column
#     record['ID'] = i
    
#     df = df.append(record, ignore_index=True)

# # Export the combined DataFrame to the same CSV file
df.to_csv('train.csv', index=False)
