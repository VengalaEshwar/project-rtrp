import numpy as np
import pandas as pd

#reading the file
def preProcess(path) :
    temp= pd.read_csv(path)
    df=temp.copy()
    df.dropna(inplace=True)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    # numeric_columns
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    X = df.drop("status", axis=1)
    y = df["status"]
    X = X.drop("name", axis=1)
    return pd.train_test_split(X, y, test_size=0.2, random_state=42)
