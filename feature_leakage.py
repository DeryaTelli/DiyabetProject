import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

df = pd.read_csv("data/diabetes.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Aynı train/test ayrımını kullan
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------
# DOĞRU YÖNTEM
# --------------------------------

correct_model = Pipeline([
    ("scaler", StandardScaler()),
    ("selection", SelectKBest(f_classif, k=10)),
    ("model", LogisticRegression(max_iter=1000))
])

correct_model.fit(X_train, y_train)

correct_pred = correct_model.predict(X_test)
correct_prob = correct_model.predict_proba(X_test)[:, 1]

# --------------------------------
# YANLIŞ YÖNTEM
# --------------------------------

# Özellik seçimi tüm veri üzerinde yapılıyor
wrong_selector = SelectKBest(f_classif, k=10)
X_selected = wrong_selector.fit_transform(X, y)

# Aynı örneklerin train/test ayrımı
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_selected, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

wrong_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

wrong_model.fit(X_train_w, y_train_w)

wrong_pred = wrong_model.predict(X_test_w)
wrong_prob = wrong_model.predict_proba(X_test_w)[:, 1]

# --------------------------------
# KARŞILAŞTIRMA
# --------------------------------

def evaluate(y_true, pred, prob):
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, prob)
    }

results = pd.DataFrame([
    {
        "Method": "Correct Feature Selection",
        **evaluate(y_test, correct_pred, correct_prob)
    },
    {
        "Method": "Leaky Feature Selection",
        **evaluate(y_test_w, wrong_pred, wrong_prob)
    }
])

print("\nFEATURE SELECTION DATA LEAKAGE")
print(results.round(4).to_string(index=False))

results.to_csv(
    "results/feature_leakage_comparison.csv",
    index=False
)
print("\nDoğru yöntemin seçtiği özellikler:")
print(
    X.columns[
        correct_model.named_steps["selection"].get_support()
    ].tolist()
)

print("\nYanlış yöntemin seçtiği özellikler:")
print(
    X.columns[
        wrong_selector.get_support()
    ].tolist()
)