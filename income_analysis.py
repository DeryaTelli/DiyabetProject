import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

os.makedirs("results", exist_ok=True)

# Veri setini yükle
df = pd.read_csv("data/diabetes.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Önceki modellerle aynı test kümesi
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Gelir kategorilerini oluştur
income_groups = pd.cut(
    X_test["Income"],
    bins=[0, 3, 6, 8],
    labels=["Dusuk", "Orta", "Yuksek"]
)

# Kaydedilmiş modeller
models = {
    "Logistic Regression + SMOTE":
        joblib.load("models/logistic_smote.joblib"),

    "Random Forest":
        joblib.load("models/random_forest.joblib")
}

results = []
fairness_results = []

for model_name, model in models.items():

    predictions = model.predict(X_test)

    # Gelir gruplarını ayrı değerlendir
    for group in ["Dusuk", "Orta", "Yuksek"]:

        mask = income_groups == group

        yt = y_test[mask]
        yp = predictions[mask]

        tn, fp, fn, tp = confusion_matrix(
            yt, yp, labels=[0, 1]
        ).ravel()

        fpr = fp / (fp + tn) if fp + tn > 0 else float("nan")
        fnr = fn / (fn + tp) if fn + tp > 0 else float("nan")

        results.append({
            "Model": model_name,
            "Income_Group": group,
            "Sample_Count": len(yt),
            "Positive_Count": int(yt.sum()),
            "Accuracy": accuracy_score(yt, yp),
            "Precision": precision_score(
                yt, yp, zero_division=0
            ),
            "Recall": recall_score(
                yt, yp, zero_division=0
            ),
            "F1": f1_score(
                yt, yp, zero_division=0
            ),
            "FPR": fpr,
            "FNR": fnr
        })

    # Gelir grupları arasındaki adalet farkları
    dpd = demographic_parity_difference(
        y_test,
        predictions,
        sensitive_features=income_groups
    )

    eod = equalized_odds_difference(
        y_test,
        predictions,
        sensitive_features=income_groups
    )

    fairness_results.append({
        "Model": model_name,
        "DPD": dpd,
        "EOD": eod
    })

results_df = pd.DataFrame(results)
fairness_df = pd.DataFrame(fairness_results)

print("\n--- INCOME GROUP PERFORMANCE ---")
print(results_df.round(4).to_string(index=False))

print("\n--- INCOME FAIRNESS ---")
print(fairness_df.round(4).to_string(index=False))

results_df.to_csv(
    "results/income_performance.csv",
    index=False
)

fairness_df.to_csv(
    "results/income_fairness.csv",
    index=False
)

print("\nGelir analizi tamamlandi.")