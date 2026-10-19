import os
import pandas as pd

os.makedirs("results", exist_ok=True)

# Önceki deneylerin sonuçları
lr = pd.read_csv(
    "results/cross_validation_results.csv"
)

rf = pd.read_csv(
    "results/random_forest_cv.csv"
)

final = pd.read_csv(
    "results/final_test_results.csv"
)

fairness = pd.read_csv(
    "results/final_fairness_metrics.csv"
)

# LR tablosuna model sütunu ekle
lr["Model"] = "Logistic Regression"

# CV tablolarını birleştir
cv_results = pd.concat(
    [lr, rf],
    ignore_index=True
)

# Sütunları düzenle
columns = [
    "Model",
    "Scenario",
    "Accuracy_Mean",
    "Precision_Mean",
    "Recall_Mean",
    "F1_Mean",
    "ROC_AUC_Mean",
    "F1_Std",
    "ROC_AUC_Std"
]

cv_results = cv_results[columns]

# Akademik tablolara hazır çıktılar
cv_results.to_csv(
    "results/paper_table_cv.csv",
    index=False
)

final.to_csv(
    "results/paper_table_final.csv",
    index=False
)

fairness.to_csv(
    "results/paper_table_fairness.csv",
    index=False
)

print("\nTABLE 1 — CROSS-VALIDATION")
print(cv_results.round(4).to_string(index=False))

print("\nTABLE 2 — FINAL TEST")
print(final.round(4).to_string(index=False))

print("\nTABLE 3 — FAIRNESS")
print(fairness.round(4).to_string(index=False))

print("\nAkademik sonuç tabloları oluşturuldu.")