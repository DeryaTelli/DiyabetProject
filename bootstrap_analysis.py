import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from fairlearn.metrics import equalized_odds_difference

os.makedirs("results", exist_ok=True)

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

# Veri seti
df = pd.read_csv("data/diabetes.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

models = {
    "Logistic Regression + SMOTE":
        joblib.load("models/logistic_smote.joblib"),
    "Random Forest":
        joblib.load("models/random_forest.joblib")
}

groups = {
    "Gender": X_test["Sex"].map({
        0: "Kadin",
        1: "Erkek"
    }),
    "Age": pd.cut(
        X_test["Age"],
        bins=[0, 3, 7, 13],
        labels=["18-34", "35-54", "55+"]
    ),
    "Income": pd.cut(
        X_test["Income"],
        bins=[0, 3, 6, 8],
        labels=["Dusuk", "Orta", "Yuksek"]
    )
}

def calculate_rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    fnr = fn / (fn + tp) if fn + tp else np.nan

    return recall, fpr, fnr


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    return tuple(np.percentile(values, [2.5, 97.5]))


group_results = []
gap_results = []

rng = np.random.default_rng(RANDOM_STATE)

for model_name, model in models.items():

    print(f"\nBootstrap: {model_name}", flush=True)

    predictions = np.asarray(model.predict(X_test))
    true_values = y_test.to_numpy()

    for group_type, group_series in groups.items():

        group_values = group_series.astype(str).to_numpy()
        unique_groups = list(dict.fromkeys(group_values))

        # Her grubun orijinal indeksleri
        group_indices = {
            g: np.where(group_values == g)[0]
            for g in unique_groups
        }

        point_rates = {}
        boot_rates = {
            g: {"Recall": [], "FPR": [], "FNR": []}
            for g in unique_groups
        }

        gap_samples = []
        eod_samples = []

        for g, indices in group_indices.items():
            recall, fpr, fnr = calculate_rates(
                true_values[indices],
                predictions[indices]
            )

            point_rates[g] = {
                "Recall": recall,
                "FPR": fpr,
                "FNR": fnr
            }

        # Bootstrap: her demografik grubun içinde
        # gözlemleri yeniden örnekle
        for _ in range(N_BOOTSTRAP):

            sampled_parts = []
            current_recalls = []
            current_fprs = []

            for g, indices in group_indices.items():

                sampled = rng.choice(
                    indices,
                    size=len(indices),
                    replace=True
                )

                sampled_parts.append(sampled)

                recall, fpr, fnr = calculate_rates(
                    true_values[sampled],
                    predictions[sampled]
                )

                boot_rates[g]["Recall"].append(recall)
                boot_rates[g]["FPR"].append(fpr)
                boot_rates[g]["FNR"].append(fnr)

                current_recalls.append(recall)
                current_fprs.append(fpr)

            if (
                np.all(np.isfinite(current_recalls))
                and np.all(np.isfinite(current_fprs))
            ):
                recall_gap = (
                    max(current_recalls)
                    - min(current_recalls)
                )

                fpr_gap = (
                    max(current_fprs)
                    - min(current_fprs)
                )

                gap_samples.append(recall_gap)
                eod_samples.append(max(recall_gap, fpr_gap))

        # Grup bazında güven aralıkları
        for g in unique_groups:

            row = {
                "Model": model_name,
                "Group_Type": group_type,
                "Group": g,
                "Sample_Count": len(group_indices[g])
            }

            for metric in ["Recall", "FPR", "FNR"]:
                low, high = confidence_interval(
                    boot_rates[g][metric]
                )

                row[metric] = point_rates[g][metric]
                row[f"{metric}_CI_Lower"] = low
                row[f"{metric}_CI_Upper"] = high

            group_results.append(row)

        gap_low, gap_high = confidence_interval(gap_samples)
        eod_low, eod_high = confidence_interval(eod_samples)

        original_recalls = [
            point_rates[g]["Recall"] for g in unique_groups
        ]

        original_fprs = [
            point_rates[g]["FPR"] for g in unique_groups
        ]

        gap_results.append({
            "Model": model_name,
            "Group_Type": group_type,
            "Recall_Gap": (
                max(original_recalls) - min(original_recalls)
            ),
            "Recall_Gap_CI_Lower": gap_low,
            "Recall_Gap_CI_Upper": gap_high,
            "EOD": max(
                max(original_recalls) - min(original_recalls),
                max(original_fprs) - min(original_fprs)
            ),
            "EOD_CI_Lower": eod_low,
            "EOD_CI_Upper": eod_high
        })

group_df = pd.DataFrame(group_results)
gap_df = pd.DataFrame(gap_results)

print("\n--- GROUP BOOTSTRAP CONFIDENCE INTERVALS ---")
print(group_df.round(4).to_string(index=False))

print("\n--- FAIRNESS GAP CONFIDENCE INTERVALS ---")
print(gap_df.round(4).to_string(index=False))

group_df.to_csv(
    "results/bootstrap_group_ci.csv",
    index=False
)

gap_df.to_csv(
    "results/bootstrap_fairness_ci.csv",
    index=False
)

print("\nBootstrap analizi tamamlandi.")