import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Veri seti
df = pd.read_csv("data/diabetes.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Önceki deneylerle aynı train/test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 2. Optimize edilmiş modeller
models = {
    "Logistic Regression + SMOTE": Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(
            k_neighbors=5,
            random_state=42
        )),
        ("model", LogisticRegression(
            C=100,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=10,
            max_depth=16,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results = []

for name, pipeline in models.items():

    print(f"\nEğitiliyor: {name}")

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, prob)
    })

    print(classification_report(
        y_test, pred, zero_division=0
    ))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        display_labels=["Negatif", "Pozitif"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    filename = (
        "logistic_smote"
        if "Logistic" in name
        else "random_forest"
    )

    plt.savefig(
        f"results/{filename}_confusion_matrix.png",
        dpi=300
    )
    plt.close()

    # Modeli kaydet
    joblib.dump(
        pipeline,
        f"models/{filename}.joblib"
    )

# 3. Karşılaştırma tablosu
results_df = pd.DataFrame(results)

print("\nFINAL TEST RESULTS")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "results/final_test_results.csv",
    index=False
)

# 4. ROC eğrileri
fig, ax = plt.subplots(figsize=(8, 6))

for name in models:
    filename = (
        "logistic_smote"
        if "Logistic" in name
        else "random_forest"
    )

    fitted_model = joblib.load(
        f"models/{filename}.joblib"
    )

    RocCurveDisplay.from_estimator(
        fitted_model,
        X_test,
        y_test,
        name=name,
        ax=ax
    )

plt.title("Final Model ROC Comparison")
plt.tight_layout()
plt.savefig("results/final_roc_comparison.png", dpi=300)
plt.close()

print("\nFinal değerlendirme tamamlandı.")