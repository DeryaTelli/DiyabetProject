import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, f_classif
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
    true_negative_rate,
)


#veri seti indirme icin 
dataset= fetch_ucirepo(id=891) 

#ozellikleri ve degiskeni ayirdik 
X = dataset.data.features
y = dataset.data.targets["Diabetes_binary"]

# tek data frame olusturduk 
df= X.copy()
df["Diabetes_binary"] = y

# veriyi bilgisayar kaydettik 
df.to_csv("data/diabetes.csv", index=False)

# veri senitini incelemek icin 
print("Veri seti boyutu:", df.shape)

print("\nİlk 5 satır:")
print(df.head())

print("\nSütun isimleri:")
print(df.columns.tolist())

print("\nEksik değer sayısı:")
print(df.isnull().sum().sum())

print("\nDiyabet sınıf dağılımı:")
print(df["Diabetes_binary"].value_counts())


#grafik olarak diyabet sınıf dağılımını göstermek için
class_counts = df["Diabetes_binary"].value_counts()

plt.bar(
    ["Diyabet Yok", "Prediyabet/Diyabet"],
    [class_counts.get(0,0), class_counts.get(1,0)]
)

plt.title("Diyabet Sınıf Dağılımı")
plt.ylabel("Sınıf Sayısı")
plt.show()


X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# veriyi %80 egitim % 30 test olarak ayirdik 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("\nEğitim verisi:", X_train.shape)
print("Test verisi:", X_test.shape)


model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

#modeli egitmek icin test verisi uzerinden 
model.fit(X_train, y_train)

#tahmin yapmak icin
y_pred = model.predict(X_test)
y_prob= model.predict_proba(X_test)[:, 1]

# performans metriklerini hesaplamak icin
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))
print("Classification Report:")
print(classification_report(y_test, y_pred))

#smote iceren model 
smote_model = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

#sadece egitim verisi uzerinden smote modelini egitmek icin
smote_model.fit(X_train, y_train)

#dokunulmamis test verisi uzerinden tahmin yapmak icin
y_pred_smote = smote_model.predict(X_test)
y_prob_smote = smote_model.predict_proba(X_test)[:, 1]

print("\nSMOTE Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print("Precision:", precision_score(y_test, y_pred_smote))
print("Recall:", recall_score(y_test, y_pred_smote))
print("F1-Score:", f1_score(y_test, y_pred_smote))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_smote))
print("Classification Report:")
print(classification_report(y_test, y_pred_smote))

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
    "Baseline Model": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob)
    ],
    "SMOTE Model": [
        accuracy_score(y_test, y_pred_smote),
        precision_score(y_test, y_pred_smote),
        recall_score(y_test, y_pred_smote),
        f1_score(y_test, y_pred_smote),
        roc_auc_score(y_test, y_prob_smote)
    ]
})

comparison["Difference"] = comparison["SMOTE Model"] - comparison["Baseline Model"]

print("\n --- MODEL COMPARISON ---")
print(comparison.round(4))

comparison.to_csv("results/smote_comparison.csv", index=False)

#ozellik secimi ve logistik regresyon modeli pipeline olusturmak icin
fs_model=ImbPipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

#modeli egitmek icin
fs_model.fit(X_train, y_train)

#test verisi uzerinden tahmin yapmak icin
y_pred_fs = fs_model.predict(X_test)
y_prob_fs = fs_model.predict_proba(X_test)[:, 1]

print("\n --- FEATURE SELECTION RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_pred_fs))
print("Precision:", precision_score(y_test, y_pred_fs))
print("Recall:", recall_score(y_test, y_pred_fs))
print("F1-Score:", f1_score(y_test, y_pred_fs))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_fs))

#secilen 10 ozelligi goruntule 
selector = fs_model.named_steps["feature_selection"]
selected_features = X_train.columns[selector.get_support()]
print("\nSeçilen 10 Özellik:")
print(selected_features.tolist())

#Feature Selection +  Smote lojistik regresyon 
fs_smote_model= ImbPipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(
        score_func=f_classif,
        k=10
    )),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

fs_smote_model.fit(X_train, y_train)

y_pred_fs_smote = fs_smote_model.predict(X_test)
y_prob_fs_smote = fs_smote_model.predict_proba(X_test)[:, 1]

print("\n --- FEATURE SELECTION + SMOTE RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_pred_fs_smote))
print("Precision:", precision_score(y_test, y_pred_fs_smote))
print("Recall:", recall_score(y_test, y_pred_fs_smote))
print("F1-Score:", f1_score(y_test, y_pred_fs_smote))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_fs_smote))

comparison= pd.DataFrame({
    "Scenario": ["Baseline Model", "SMOTE Model", "Feature Selection Model", "Feature Selection + SMOTE Model"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_smote),
        accuracy_score(y_test, y_pred_fs),
        accuracy_score(y_test, y_pred_fs_smote)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, y_pred_smote),
        recall_score(y_test, y_pred_fs),
        recall_score(y_test, y_pred_fs_smote)
    ],
    "F1-Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, y_pred_smote),
        f1_score(y_test, y_pred_fs),
        f1_score(y_test, y_pred_fs_smote)
    ],
    "AUC-ROC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, y_prob_smote),
        roc_auc_score(y_test, y_prob_fs),
        roc_auc_score(y_test, y_prob_fs_smote)
    ]
})

print("\n --- FINAL MODEL COMPARISON ---")
print(comparison.round(4))

comparison.to_csv("results/final_model_comparison.csv", index=False)

#test veri setinin cinsiyet bilgilerini al 
sex_test= X_test["Sex"]

# degerlendirme metrikleri 
fairness_metrics= {
    "Accuracy": accuracy_score,
    "Precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
    "Recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
    "F1-Score": lambda yt, yp: f1_score(yt, yp, zero_division=0),
}

#dort modelimiz bir sozlukte topladik 
# Modelleri tanımla
models = {
    "Baseline": model,
    "SMOTE": smote_model,
    "Feature Selection": fs_model,
    "Feature Selection + SMOTE": fs_smote_model
}

gender_results = []
gender_fairness = []

for model_name, trained_model in models.items():

    # Test tahminlerini al
    predictions = trained_model.predict(X_test)

    # Her cinsiyet grubu için metrikleri hesapla
    frame = MetricFrame(
        metrics=fairness_metrics,
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=X_test["Sex"]
    )

    # Sonuç tablosunu oluştur
    group_table = frame.by_group.reset_index()
    group_table["Model"] = model_name

    # ÖNEMLİ: Listeye ekle
    gender_results.append(group_table)

    # Fairness metriklerini hesapla
    dpd = demographic_parity_difference(
        y_test,
        predictions,
        sensitive_features=X_test["Sex"]
    )

    eod = equalized_odds_difference(
        y_test,
        predictions,
        sensitive_features=X_test["Sex"]
    )

    gender_fairness.append({
        "Model": model_name,
        "Demographic_Parity_Difference": dpd,
        "Equalized_Odds_Difference": eod
    })

# Döngü bittikten sonra tabloları birleştir
gender_df = pd.concat(
    gender_results,
    ignore_index=True
)

fairness_df = pd.DataFrame(gender_fairness)

gender_df["Sex"] = gender_df["Sex"].map({
    0: "Kadın",
    1: "Erkek"
})

print("\n--- CİNSİYETE GÖRE PERFORMANS ---")
print(gender_df.round(4).to_string(index=False))

print("\n--- FAIRNESS METRİKLERİ ---")
print(fairness_df.round(4).to_string(index=False))

# Yaş gruplarını oluştur
age_groups = pd.cut(
    X_test["Age"],
    bins=[0, 3, 7, 13],
    labels=["18-34", "35-54", "55+"]
)

age_results = []
age_fairness_results = []

for model_name, trained_model in models.items():

    predictions = trained_model.predict(X_test)

    frame = MetricFrame(
        metrics=fairness_metrics,
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=age_groups
    )

    group_table = frame.by_group.reset_index()
    group_table["Model"] = model_name

    age_results.append(group_table)

    age_fairness_results.append({
        "Model": model_name,
        "Demographic_Parity_Difference":
            demographic_parity_difference(
                y_test,
                predictions,
                sensitive_features=age_groups
            ),
        "Equalized_Odds_Difference":
            equalized_odds_difference(
                y_test,
                predictions,
                sensitive_features=age_groups
            )
    })

age_df = pd.concat(age_results, ignore_index=True)
age_fairness_df = pd.DataFrame(age_fairness_results)

print("\n--- YAŞ GRUPLARINA GÖRE PERFORMANS ---")
print(age_df.round(4).to_string(index=False))

print("\n--- YAŞ GRUPLARI FAIRNESS METRİKLERİ ---")
print(age_fairness_df.round(4).to_string(index=False))

plt.figure(figsize=(11, 6))

sns.barplot(
    data=gender_df,
    x="Model",
    y="Recall",
    hue="Sex"
)

plt.title("Cinsiyete Göre Recall Karşılaştırması")
plt.ylabel("Recall")
plt.xlabel("Model")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "results/gender_recall.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(11, 6))

sns.barplot(
    data=age_df,
    x="Model",
    y="Recall",
    hue=age_df.columns[0]
)

plt.title("Yaş Gruplarına Göre Recall Karşılaştırması")
plt.ylabel("Recall")
plt.xlabel("Model")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "results/age_recall.png",
    dpi=300
)

plt.show()

gender_df.to_csv(
    "results/gender_performance.csv",
    index=False
)

fairness_df.to_csv(
    "results/gender_fairness.csv",
    index=False
)

age_df.to_csv(
    "results/age_performance.csv",
    index=False
)

age_fairness_df.to_csv(
    "results/age_fairness.csv",
    index=False
)

print("Fairness analizleri kaydedildi.")
