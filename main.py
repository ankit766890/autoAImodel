from utils import load_data, preprocess_data, train_models, evaluate_models
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

df = load_data("sample_dataset.csv")
X_train, X_test, y_train, y_test = preprocess_data(df)
models = train_models(X_train, y_train)
evaluate_models(models, X_test, y_test)

# Cross-validation ke liye label encode karein
X = df.drop('species', axis=1)
le = LabelEncoder()
y = le.fit_transform(df['species'])

print("\nCross Validation Results:")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean()*100:.2f}% average accuracy (CV)")