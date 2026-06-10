# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:38:17 2026

@author: menci
"""

"""
plot_distribution.py
--------------------
Genera una gráfica de distribución por clase del split de Train
leyendo directamente las etiquetas YOLO del dataset balanceado.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# =============================================================================
# CONFIG
# =============================================================================

DATASET_DIR = "C:/Users/menci/TFGMencia/ModificacionesDataset/datasetBalanceado-v3"
OUTPUT_IMG  = "C:/Users/menci/TFGMencia/ModificacionesDataset/distribucion_train_balanceado.png"

# =============================================================================


def count_instances(split_dir):
    """Cuenta instancias por clase en un split leyendo los .txt de etiquetas."""
    labels_dir = os.path.join(split_dir, "labels")
    counts = defaultdict(int)
    if not os.path.exists(labels_dir):
        return counts
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, fname), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    class_id = int(line.split()[0])
                    counts[class_id] += 1
    return counts


def load_class_names(dataset_dir):
    """Lee los nombres de clases del data.yaml."""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    names = {}
    if not os.path.exists(yaml_path):
        return names
    with open(yaml_path, "r") as f:
        content = f.read()

    in_names = False
    idx = 0
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            if "[" in stripped:
                inner = stripped.split("[")[1].split("]")[0]
                for i, n in enumerate(inner.split(",")):
                    names[i] = n.strip().strip("'\"")
                break
            continue
        if in_names:
            if stripped.startswith("-"):
                names[idx] = stripped.lstrip("- ").strip().strip("'\"")
                idx += 1
            elif stripped and not stripped.startswith("#"):
                break
    return names


def main():
    # Contar solo train
    train_dir = os.path.normpath(os.path.join(DATASET_DIR, "train"))
    train_counts = count_instances(train_dir)

    # Nombres de clases
    class_names = load_class_names(DATASET_DIR)

    # Ordenar por cantidad descendente
    all_classes = sorted(train_counts.keys(), key=lambda c: train_counts[c], reverse=True)
    class_labels = [class_names.get(c, str(c)) for c in all_classes]
    counts = [train_counts[c] for c in all_classes]

    x = np.arange(len(all_classes))

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.bar(x, counts, width=0.6, color="#2ecc71", alpha=0.9, label="Training")

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Class Name", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribución por clase (Train) — Dataset Balanceado", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    plt.show()
    print(f"Gráfica guardada en: {OUTPUT_IMG}")


if __name__ == "__main__":
    main()