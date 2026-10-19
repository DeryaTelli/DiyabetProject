import os
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

os.makedirs("results", exist_ok=True)

# Veri seti
df = pd.read_csv("data/diabetes.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Aynı train/test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 5-Fold CV
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

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

scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC_AUC": "roc_auc"
}

results = []

for scenario_name, steps in scenarios.items():

    print(f"Çalışıyor: {scenario_name}", flush=True)

    pipeline = Pipeline([
        *steps,
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
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

    row = {
        "Model": "Random Forest",
        "Scenario": scenario_name
    }

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric}_Mean"] = values.mean()
        row[f"{metric}_Std"] = values.std()

    results.append(row)

results_df = pd.DataFrame(results)

print("\nRANDOM FOREST CV RESULTS")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "results/random_forest_cv.csv",
    index=False
)