import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="Set2")

df = pd.read_csv("datos.csv", sep=";")


#Diccionario para mapear el número correspondiente a las clase al nombre de esa
class_mapping = {
    0: "Apple",
    1: "Orange",
    2: "Tomato",
    3: "Peach",
    4: "Quince",
    5: "Pomegranate",
    6: "Pear",
    7: "Lemon",
    8: "Rice",
    9: "Plum",
    10: "DragonFruit",
    11: "bellPepper",
    12: "Saurieng",
    13: "Aubegine",
    14: "Radish",
    15: "Cherry",
    16: "Apricot",
    17: "Courgette",
    18: "Bean",
    19: "Carrot",
    20: "Mango",
    21: "Strawberry",
    22: "Salad",
    23: "Banana",
    24: "Fig",
    25: "Grape",
    26: "Durian",
    27: "Pineapple",
    28: "Watermelon",
    29: "Lime",
    30: "Lychee"
}

df["Class_ID"] = df["Class Name"]

df["Class Name"] = df["Class_ID"].map(class_mapping)

df = df.dropna(subset=["Class Name"])

df = df.sort_values("Class_ID")

df = df.groupby(["Class_ID", "Class Name"], as_index=False).sum()

num_classes = len(df)
x = np.arange(num_classes)

plt.figure(figsize=(max(16, num_classes * 0.6), 8))

plt.bar(x, df["Training Count"], label="Training")
plt.bar(x, df["Validation Count"], bottom=df["Training Count"], label="Validation")
plt.bar(
    x,
    df["Test Count"],
    bottom=df["Training Count"] + df["Validation Count"],
    label="Test"
)

plt.xlabel("Class Name")
plt.ylabel("Count")
plt.title("Distribución por clase (Train / Validation / Test)")

plt.xticks(x, df["Class Name"], rotation=45, ha="right", fontsize=9)

plt.legend()
plt.tight_layout()
plt.show()

totals = df[[
    "Training Count",
    "Validation Count",
    "Test Count"
]].sum(axis=1)

train_prop = df["Training Count"] / totals
val_prop = df["Validation Count"] / totals
test_prop = df["Test Count"] / totals

plt.figure(figsize=(max(16, num_classes * 0.6), 8))

plt.bar(x, train_prop, label="Training")
plt.bar(x, val_prop, bottom=train_prop, label="Validation")
plt.bar(x, test_prop, bottom=train_prop + val_prop, label="Test")

plt.xlabel("Class Name")
plt.ylabel("Proportion")
plt.title("Proporción por clase (Train / Validation / Test)")

plt.xticks(x, df["Class Name"], rotation=45, ha="right", fontsize=9)

plt.legend()
plt.tight_layout()
plt.show()