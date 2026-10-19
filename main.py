import pandas as pd
import matplotlib.pyplot as plt

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