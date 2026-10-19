import os
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

os.makedirs("results", exist_ok=True)

# Veri setini yükle
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

# 3-Fold CV (arama süresini azaltmak için)
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# ---------------------------------
# 1. Logistic Regression + SMOTE
# ---------------------------------

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

lr_params = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__class_weight": [None, "balanced"],
    "smote__k_neighbors": [3, 5, 7]
}

lr_search = RandomizedSearchCV(
    estimator=lr_pipeline,
    param_distributions=lr_params,
    n_iter=10,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nLogistic Regression tuning başladı...")
lr_search.fit(X_train, y_train)

print("En iyi parametreler:", lr_search.best_params_)
print("En iyi CV F1:", lr_search.best_score_)

# ---------------------------------
# 2. Random Forest Baseline
# ---------------------------------

rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=1
    ))
])

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [8, 12, 16, None],
    "model__min_samples_leaf": [2, 5, 10],
    "model__class_weight": [None, "balanced"]
}

rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_params,
    n_iter=10,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nRandom Forest tuning başladı...")
rf_search.fit(X_train, y_train)

print("En iyi parametreler:", rf_search.best_params_)
print("En iyi CV ROC-AUC:", rf_search.best_score_)

# Sonuçları kaydet
summary = pd.DataFrame([
    {
        "Model": "Logistic Regression + SMOTE",
        "Optimization_Metric": "F1",
        "Best_CV_Score": lr_search.best_score_,
        "Best_Params": str(lr_search.best_params_)
    },
    {
        "Model": "Random Forest Baseline",
        "Optimization_Metric": "ROC-AUC",
        "Best_CV_Score": rf_search.best_score_,
        "Best_Params": str(rf_search.best_params_)
    }
])

summary.to_csv(
    "results/hyperparameter_tuning.csv",
    index=False
)

print("\nHiperparametre optimizasyonu tamamlandı.")