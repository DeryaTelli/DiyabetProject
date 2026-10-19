import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

os.makedirs("results", exist_ok=True)

# Kaydettiğimiz veri setini oku
df = pd.read_csv("data/diabetes.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# ÖNCE veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# DOĞRU: SMOTE yalnızca train üzerinde
correct_model = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

correct_model.fit(X_train, y_train)

correct_pred = correct_model.predict(X_test)
correct_prob = correct_model.predict_proba(X_test)[:, 1]

# YANLIŞ yöntem deneme amaclı 

# Tüm veriden ölçeklendirme öğreniliyor
wrong_scaler = StandardScaler()
X_scaled = wrong_scaler.fit_transform(X)

# Tüm veri setine SMOTE uygulanıyor
wrong_smote = SMOTE(random_state=42)

X_resampled, y_resampled = wrong_smote.fit_resample(
    X_scaled, y
)

# SMOTE sonrası veriyi ayırıyoruz (YANLIŞ)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.20,
    random_state=42,
    stratify=y_resampled
)

wrong_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

wrong_model.fit(X_train_w, y_train_w)

wrong_pred = wrong_model.predict(X_test_w)
wrong_prob = wrong_model.predict_proba(X_test_w)[:, 1]

def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    }


results = pd.DataFrame([
    {
        "Method": "Correct Pipeline",
        **evaluate(y_test, correct_pred, correct_prob)
    },
    {
        "Method": "Wrong Preprocessing",
        **evaluate(y_test_w, wrong_pred, wrong_prob)
    }
])

print("\nDATA LEAKAGE COMPARISON")
print(results.round(4).to_string(index=False))

results.to_csv(
    "results/data_leakage_comparison.csv",
    index=False
)

plot_data = results.set_index("Method")[
    ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
].T

ax = plot_data.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Correct vs Incorrect Preprocessing")
plt.ylabel("Metric Score")
plt.xlabel("Evaluation Metric")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Method")
plt.tight_layout()

plt.savefig(
    "results/data_leakage_comparison.png",
    dpi=300
)

plt.show()