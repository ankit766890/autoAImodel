import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.dropna()
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    X = df.drop('species', axis=1)
    y = df['species']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss')
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        score = model.score(X_test, y_test)
        print(f"{name}: {score*100:.2f}% accuracy")