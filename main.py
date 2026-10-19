import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo


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

class_counts = df["Diabetes_binary"].value_counts()

plt.bar(
    ["Diyabet Yok", "Prediyabet/Diyabet"],
    [class_counts.get(0,0), class_counts.get(1,0)]
)

plt.title("Diyabet Sınıf Dağılımı")
plt.ylabel("Sınıf Sayısı")
plt.show()



