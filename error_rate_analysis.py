import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

os.makedirs("results", exist_ok=True)

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

# Kaydedilmiş modeller
models = {
    "Logistic Regression + SMOTE":
        joblib.load("models/logistic_smote.joblib"),
    "Random Forest":
        joblib.load("models/random_forest.joblib")
}

# Demografik gruplar
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

results = []

for model_name, model in models.items():

    predictions = pd.Series(
        model.predict(X_test),
        index=X_test.index
    )

    for group_type, group_values in groups.items():

        for group in group_values.dropna().unique():

            mask = group_values == group

            yt = y_test.loc[mask]
            yp = predictions.loc[mask]

            tn, fp, fn, tp = confusion_matrix(
                yt, yp, labels=[0, 1]
            ).ravel()

            fpr = (
                fp / (fp + tn)
                if (fp + tn) > 0 else float("nan")
            )

            fnr = (
                fn / (fn + tp)
                if (fn + tp) > 0 else float("nan")
            )

            results.append({
                "Model": model_name,
                "Group_Type": group_type,
                "Group": str(group),
                "N": len(yt),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "FPR": fpr,
                "FNR": fnr
            })

results_df = pd.DataFrame(results)

print("\n--- FNR/FPR RESULTS ---")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "results/fnr_fpr_results.csv",
    index=False
)

# Grafikler
for group_type in groups:

    plot_df = results_df[
        results_df["Group_Type"] == group_type
    ]

    long_df = plot_df.melt(
        id_vars=["Model", "Group"],
        value_vars=["FNR", "FPR"],
        var_name="Metric",
        value_name="Rate"
    )

    g = sns.catplot(
        data=long_df,
        x="Group",
        y="Rate",
        hue="Model",
        col="Metric",
        kind="bar",
        height=4,
        aspect=1.1
    )

    g.set_axis_labels("Group", "Error Rate")
    g.set(ylim=(0, 1))

    g.fig.savefig(
        f"results/{group_type.lower()}_error_rates.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(g.fig)

print("\nFNR/FPR analizi tamamlandi.")