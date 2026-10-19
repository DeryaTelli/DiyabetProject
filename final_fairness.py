import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference
)

os.makedirs("results", exist_ok=True)

# 1. Veri setini yükle
df = pd.read_csv("data/diabetes.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Önceki deneylerle aynı test kümesini oluştur
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 2. Kaydedilmiş modelleri yükle
models = {
    "Logistic Regression + SMOTE":
        joblib.load("models/logistic_smote.joblib"),

    "Random Forest":
        joblib.load("models/random_forest.joblib")
}

# 3. Fairness metrikleri
metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda yt, yp: precision_score(
        yt, yp, zero_division=0
    ),
    "Recall": lambda yt, yp: recall_score(
        yt, yp, zero_division=0
    ),
    "F1": lambda yt, yp: f1_score(
        yt, yp, zero_division=0
    )
}

# 4. Demografik gruplar
gender = X_test["Sex"].map({
    0: "Kadın",
    1: "Erkek"
}).rename("Gender")

# Age sütunu gerçek yaş değil, yaş kategorisi
age = pd.cut(
    X_test["Age"],
    bins=[0, 3, 7, 13],
    labels=["18-34", "35-54", "55+"]
).rename("Age_Group")

group_results = []
fairness_results = []

# 5. Her modeli değerlendir
for model_name, model in models.items():

    print(f"\nAnaliz ediliyor: {model_name}")

    predictions = model.predict(X_test)

    for group_name, sensitive in {
        "Gender": gender,
        "Age": age
    }.items():

        frame = MetricFrame(
            metrics=metrics,
            y_true=y_test,
            y_pred=predictions,
            sensitive_features=sensitive
        )

        group_table = frame.by_group.reset_index()
        group_table = group_table.rename(
            columns={group_table.columns[0]: "Group"}
        )

        # Gruplardaki toplam ve pozitif örnek sayıları
        counts = pd.DataFrame({
            "Group": sensitive,
            "Target": y_test
        }).groupby("Group", observed=True)["Target"].agg(
            Sample_Count="size",
            Positive_Count="sum"
        ).reset_index()

        group_table = group_table.merge(
            counts, on="Group", how="left"
        )

        group_table["Model"] = model_name
        group_table["Group_Type"] = group_name

        group_results.append(group_table)

        # Fairness farklarını hesapla
        dpd = demographic_parity_difference(
            y_test,
            predictions,
            sensitive_features=sensitive
        )

        eod = equalized_odds_difference(
            y_test,
            predictions,
            sensitive_features=sensitive
        )

        recall_gap = (
            frame.by_group["Recall"].max()
            - frame.by_group["Recall"].min()
        )

        fairness_results.append({
            "Model": model_name,
            "Group_Type": group_name,
            "DPD": dpd,
            "EOD": eod,
            "Recall_Gap": recall_gap
        })

# 6. Sonuç tabloları
group_df = pd.concat(
    group_results,
    ignore_index=True
)

fairness_df = pd.DataFrame(fairness_results)

print("\n--- GRUP BAZLI PERFORMANS ---")
print(group_df.round(4).to_string(index=False))

print("\n--- FAIRNESS KARŞILAŞTIRMASI ---")
print(fairness_df.round(4).to_string(index=False))

# 7. CSV olarak kaydet
group_df.to_csv(
    "results/final_group_performance.csv",
    index=False
)

fairness_df.to_csv(
    "results/final_fairness_metrics.csv",
    index=False
)

print("\nFairness analizleri tamamlandı.")

# Cinsiyet ve yaş için ayrı Recall grafikleri

for group_type in ["Gender", "Age"]:

    filtered = group_df[
        group_df["Group_Type"] == group_type
    ]

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=filtered,
        x="Group",
        y="Recall",
        hue="Model"
    )

    plt.title(f"{group_type} - Recall Comparison")
    plt.ylabel("Recall")
    plt.xlabel("Group")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(
        f"results/final_{group_type.lower()}_recall.png",
        dpi=300
    )

    plt.close()

# Fairness farklarının karşılaştırması
fairness_plot = fairness_df.melt(
    id_vars=["Model", "Group_Type"],
    value_vars=["DPD", "EOD", "Recall_Gap"],
    var_name="Metric",
    value_name="Difference"
)

g = sns.catplot(
    data=fairness_plot,
    x="Metric",
    y="Difference",
    hue="Model",
    col="Group_Type",
    kind="bar",
    height=5,
    aspect=1.1
)

g.fig.savefig(
    "results/final_fairness_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")