import pandas as pd
from joblib import load

# Load the trained models
rfc = load('random_forest_model.joblib')
svc = load('svc_model.joblib')
gbc = load('gradient_boosting_model.joblib')
knn = load('knn_model.joblib')

# Define function to take user input for test case
def get_test_case_from_user():
    test_case = {}
    print("Enter the following features for your test case:")
    test_case['MDVP:Fo(Hz)'] = [float(input("MDVP:Fo(Hz): "))]
    test_case['MDVP:Fhi(Hz)'] = [float(input("MDVP:Fhi(Hz): "))]
    test_case['MDVP:Flo(Hz)'] = [float(input("MDVP:Flo(Hz): "))]
    test_case['MDVP:Jitter(%)'] = [float(input("MDVP:Jitter(%): "))]
    test_case['MDVP:Jitter(Abs)'] = [float(input("MDVP:Jitter(Abs): "))]
    test_case['MDVP:RAP'] = [float(input("MDVP:RAP: "))]
    test_case['MDVP:PPQ'] = [float(input("MDVP:PPQ: "))]
    test_case['Jitter:DDP'] = [float(input("Jitter:DDP: "))]
    test_case['MDVP:Shimmer'] = [float(input("MDVP:Shimmer: "))]
    test_case['MDVP:Shimmer(dB)'] = [float(input("MDVP:Shimmer(dB): "))]
    test_case['Shimmer:APQ3'] = [float(input("Shimmer:APQ3: "))]
    test_case['Shimmer:APQ5'] = [float(input("Shimmer:APQ5: "))]
    test_case['MDVP:APQ'] = [float(input("MDVP:APQ: "))]
    test_case['Shimmer:DDA'] = [float(input("Shimmer:DDA: "))]
    test_case['NHR'] = [float(input("NHR: "))]
    test_case['HNR'] = [float(input("HNR: "))]
    test_case['RPDE'] = [float(input("RPDE: "))]
    test_case['DFA'] = [float(input("DFA: "))]
    test_case['spread1'] = [float(input("spread1: "))]
    test_case['spread2'] = [float(input("spread2: "))]
    test_case['D2'] = [float(input("D2: "))]
    test_case['PPE'] = [float(input("PPE: "))]
    return test_case

# Get test case from user
test_case = get_test_case_from_user()

# Convert test case to DataFrame
test_df = pd.DataFrame(test_case)

# Predict status using the loaded models
rfc_prediction = rfc.predict(test_df)
svc_prediction = svc.predict(test_df)
gbc_prediction = gbc.predict(test_df)
knn_prediction = knn.predict(test_df)

print("Random Forest Classifier Prediction:", rfc_prediction[0])
print("SVC Prediction:", svc_prediction[0])
print("Gradient Boosting Classifier Prediction:", gbc_prediction[0])
print("K-Nearest Neighbors Classifier Prediction:", knn_prediction[0])
