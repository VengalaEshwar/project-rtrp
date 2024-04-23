import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load  

df = pd.read_csv("parkinsons.csv")
df.isnull().sum()
df.dropna(inplace=True)
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

X = df.drop("status", axis=1)
y = df["status"]
X = X.drop("name", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")

# Save Random Forest Classifier model
dump(rfc, 'random_forest_model.joblib')

svc = SVC(random_state=42)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"SVC Accuracy: {accuracy_svc}")

# Save SVC model
dump(svc, 'svc_model.joblib')

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
print(f"Gradient Boosting Classifier Accuracy: {accuracy_gbc}")

# Save Gradient Boosting Classifier model
dump(gbc, 'gradient_boosting_model.joblib')

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K-Nearest Neighbors Classifier Accuracy: {accuracy_knn}")

# Save K-Nearest Neighbors Classifier model
dump(knn, 'knn_model.joblib')
