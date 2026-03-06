import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.dropna()
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y
