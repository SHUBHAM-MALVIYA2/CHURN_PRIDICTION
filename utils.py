import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.dropna()
    for col in df.select_dtypes(include='object'):
        df[col] = LabelEncoder().fit_transform(df[col])
    return df