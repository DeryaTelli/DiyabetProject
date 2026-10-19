import os
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

os.makedirs("results", exist_ok=True)

# 1. Veri setini yükle
df = pd.read_csv("data/diabetes.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# 2. Orijinal test kümesini ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Stratified 5-Fold
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 4. Deney senaryoları
scenarios = {
    "Baseline": [],

    "SMOTE": [
        ("smote", SMOTE(random_state=42))
    ],

    "Feature Selection": [
        ("selection", SelectKBest(f_classif, k=10))
    ],

    "Feature Selection + SMOTE": [
        ("selection", SelectKBest(f_classif, k=10)),
        ("smote", SMOTE(random_state=42))
    ]
}

# 5. Ölçülecek metrikler
scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC_AUC": "roc_auc"
}

results = []

for scenario_name, steps in scenarios.items():

    print(f"\nÇalışıyor: {scenario_name}", flush=True)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        *steps,
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    row = {"Scenario": scenario_name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric}_Mean"] = values.mean()
        row[f"{metric}_Std"] = values.std()

    results.append(row)

# 6. Sonuçları tabloya dönüştür
results_df = pd.DataFrame(results)

print("\nCROSS-VALIDATION RESULTS")
print(results_df.round(4).to_string(index=False))

# 7. CSV olarak kaydet
results_df.to_csv(
    "results/cross_validation_results.csv",
    index=False
)

print("\nSonuçlar kaydedildi.")