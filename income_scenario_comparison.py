
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# =====================================
# 1. VERI SETINI YUKLE
# =====================================

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/diabetes.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# =====================================
# 2. DEMOGRAFIK GRUPLARI OLUSTUR
# =====================================

group_definitions = {
    "Income": pd.cut(
        X_test["Income"],
        bins=[0, 3, 6, 8],
        labels=["Dusuk", "Orta", "Yuksek"]
    ),

    "Gender": X_test["Sex"].map({
        0: "Kadin",
        1: "Erkek"
    }),

    "Age": pd.cut(
        X_test["Age"],
        bins=[0, 3, 7, 13],
        labels=["18-34", "35-54", "55+"]
    )
}


# =====================================
# 3. DENEY SENARYOLARI
# =====================================

scenarios = [
    "Baseline",
    "SMOTE",
    "Feature Selection",
    "Feature Selection + SMOTE"
]

model_names = [
    "Logistic Regression",
    "Random Forest"
]


def create_pipeline(model_name, scenario):

    steps = []

    if model_name == "Logistic Regression":

        steps.append((
            "scaler",
            StandardScaler()
        ))

        classifier = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

    elif model_name == "Random Forest":

        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError("Bilinmeyen model")

    if "Feature Selection" in scenario:
        steps.append((
            "selection",
            SelectKBest(
                score_func=f_classif,
                k=10
            )
        ))

    if "SMOTE" in scenario:
        steps.append((
            "smote",
            SMOTE(random_state=42)
        ))

    steps.append(("model", classifier))

    return Pipeline(steps)


# =====================================
# 4. MODELLERI EGIT VE DEGERLENDIR
# =====================================

results = []

for model_name in model_names:

    for scenario in scenarios:

        print(
            f"Calisiyor: {model_name} - {scenario}",
            flush=True
        )

        pipeline = create_pipeline(
            model_name,
            scenario
        )

        # Sadece egitim verisine fit et
        pipeline.fit(X_train, y_train)

        # Orijinal test verisinde tahmin
        predictions = pd.Series(
            pipeline.predict(X_test),
            index=X_test.index
        )

        for group_type, group_series in group_definitions.items():

            # Mevcut grup isimlerini al
            group_names = group_series.dropna().unique()

            for group in group_names:

                mask = group_series == group

                yt = y_test.loc[mask]
                yp = predictions.loc[mask]

                tn, fp, fn, tp = confusion_matrix(
                    yt,
                    yp,
                    labels=[0, 1]
                ).ravel()

                # False Negative Rate
                fnr = (
                    fn / (fn + tp)
                    if (fn + tp) > 0
                    else np.nan
                )

                # False Positive Rate
                fpr = (
                    fp / (fp + tn)
                    if (fp + tn) > 0
                    else np.nan
                )

                recall = (
                    tp / (tp + fn)
                    if (tp + fn) > 0
                    else np.nan
                )

                results.append({
                    "Model": model_name,
                    "Scenario": scenario,
                    "Group_Type": group_type,
                    "Group": str(group),
                    "N": len(yt),
                    "Positive_Count": int(yt.sum()),
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                    "FNR": fnr,
                    "FPR": fpr,
                    "Recall": recall
                })


# =====================================
# 5. SONUCLARI TABLOYA DONUSTUR
# =====================================

results_df = pd.DataFrame(results)

# Baseline senaryosunu ayir
baseline = (
    results_df.loc[
        results_df["Scenario"] == "Baseline",
        [
            "Model",
            "Group_Type",
            "Group",
            "FNR",
            "FPR"
        ]
    ]
    .rename(columns={
        "FNR": "Baseline_FNR",
        "FPR": "Baseline_FPR"
    })
)

# Her grubu ayni modelin baseline
# sonucu ile eslestir
results_df = results_df.merge(
    baseline,
    on=["Model", "Group_Type", "Group"],
    how="left",
    validate="many_to_one"
)

# Degisimleri hesapla
results_df["Delta_FNR"] = (
    results_df["FNR"]
    - results_df["Baseline_FNR"]
)

results_df["Delta_FPR"] = (
    results_df["FPR"]
    - results_df["Baseline_FPR"]
)


# =====================================
# 6. TABLOLARI YAZDIR
# =====================================

for group_type in ["Income", "Gender", "Age"]:

    filtered = results_df[
        results_df["Group_Type"] == group_type
    ]

    print(
        f"\n--- {group_type.upper()} SCENARIO COMPARISON ---"
    )

    columns = [
        "Model",
        "Scenario",
        "Group",
        "N",
        "FNR",
        "FPR",
        "Recall",
        "Delta_FNR",
        "Delta_FPR"
    ]

    print(
        filtered[columns]
        .round(4)
        .to_string(index=False)
    )


# =====================================
# 7. CSV DOSYALARINI KAYDET
# =====================================

results_df.to_csv(
    "results/all_subgroup_scenario_comparison.csv",
    index=False
)

for group_type in ["Income", "Gender", "Age"]:

    filtered = results_df[
        results_df["Group_Type"] == group_type
    ]

    filtered.to_csv(
        f"results/{group_type.lower()}_scenario_comparison.csv",
        index=False
    )


# =====================================
# 8. GRAFIKLERI OLUSTUR
# =====================================

sns.set_theme(style="whitegrid")

group_orders = {
    "Income": ["Dusuk", "Orta", "Yuksek"],
    "Gender": ["Kadin", "Erkek"],
    "Age": ["18-34", "35-54", "55+"]
}

for group_type in ["Income", "Gender", "Age"]:

    filtered = results_df[
        results_df["Group_Type"] == group_type
    ]

    for metric in ["FNR", "FPR"]:

        g = sns.catplot(
            data=filtered,
            x="Group",
            y=metric,
            hue="Scenario",
            col="Model",
            kind="bar",
            order=group_orders[group_type],
            hue_order=scenarios,
            height=5,
            aspect=1.1,
            errorbar=None
        )

        g.set_axis_labels(
            f"{group_type} Group",
            metric
        )

        g.set(ylim=(0, 1))

        g.fig.suptitle(
            f"{group_type} - {metric} Comparison",
            y=1.05
        )

        g.fig.savefig(
            f"results/{group_type.lower()}_scenario_{metric.lower()}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(g.fig)


# =====================================
# 9. KONTROL VE BITIS
# =====================================

expected_rows = 2 * 4 * (3 + 2 + 3)

assert len(results_df) == expected_rows, (
    f"Beklenen {expected_rows} satir, "
    f"bulunan {len(results_df)}"
)

assert results_df[
    ["Baseline_FNR", "Baseline_FPR"]
].notna().all().all(), "Baseline eslestirmesi eksik!"

print("\n====================================")
print("TUM ALT GRUP ANALIZLERI TAMAMLANDI")
print("====================================")

print("Toplam sonuc satiri:", len(results_df))
print("CSV ve grafikler results/ klasorune kaydedildi.")
