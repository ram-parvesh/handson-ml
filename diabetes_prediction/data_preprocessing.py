import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(test_size=0.2, random_state=42):
    # Load the dataset
    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_path = BASE_DIR / "datasets" / "diabetes.csv"
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df =df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})  # Rename the column for consistency
    cols_with_zeroes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeroes] = df[cols_with_zeroes].replace(0, np.nan)  # Replace zeroes with NaN for specific columns

    df["Glucose"] = df["Glucose"].fillna(df["Glucose"].mean())
    df["BloodPressure"] = df["BloodPressure"].fillna(df["BloodPressure"].mean())
    df["SkinThickness"] = df["SkinThickness"].fillna(df["SkinThickness"].median())
    df["Insulin"] = df["Insulin"].fillna(df["Insulin"].median())
    df["BMI"] = df["BMI"].fillna(df["BMI"].median())

    # Split features and target variable
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale the features
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test