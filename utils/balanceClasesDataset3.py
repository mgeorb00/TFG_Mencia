"""
balance_dataset.py
------------------
Undersampling de un dataset YOLO (estructura Roboflow) para limitar
el número de instancias por clase a un máximo configurable.

Uso:
    python balance_dataset.py
"""

import os
import random
import shutil
from collections import defaultdict

# =============================================================================
# CONFIG
# =============================================================================

DATASET_DIR = "C:/Users/menci/TFGMencia/ModificacionesDataset/DatasetTFG-2502.v2i.yolov11"
OUTPUT_DIR  = "C:/Users/menci/TFGMencia/ModificacionesDataset/datasetBalanceado-v3"
MAX_INSTANCES = 200
RANDOM_SEED   = 42
SPLITS = ["train", "valid", "test"]

# =============================================================================


def count_instances_in_file(label_path):
    """Devuelve un dict con las clases que aparecen en un .txt de YOLO."""
    counts = defaultdict(int)
    if not os.path.exists(label_path):
        return counts
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                class_id = int(line.split()[0])
                counts[class_id] += 1
    return counts


def get_all_images(split_dir):
    """Devuelve lista de rutas normalizadas de imágenes en split_dir/images/."""
    images_dir = os.path.join(split_dir, "images")
    if not os.path.exists(images_dir):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [
        os.path.normpath(os.path.join(images_dir, f))
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]


def label_path_from_image(image_path):
    """Dado un path de imagen devuelve el path de su .txt de etiqueta."""
    image_path = os.path.normpath(image_path)
    parts = image_path.split(os.sep)
    parts[parts.index("images")] = "labels"
    label = os.path.splitext(os.sep.join(parts))[0] + ".txt"
    return label


def select_images(image_paths, max_instances, seed):
    """
    Selecciona un subconjunto de imágenes de forma que ninguna clase
    supere max_instances instancias en total.

    Estrategia:
      1. Filtra imágenes que no existen en disco.
      2. Mezcla aleatoriamente.
      3. Va añadiendo imágenes mientras ninguna clase supere el límite.
      4. Si una imagen haría superar el límite en alguna clase, se descarta.
    """
    random.seed(seed)

    # Filtrar archivos que no existen
    existing = []
    missing = 0
    for p in image_paths:
        if os.path.exists(p):
            existing.append(p)
        else:
            missing += 1

    if missing > 0:
        print(f"    AVISO: {missing} imágenes no encontradas en disco, se ignoran.")

    random.shuffle(existing)

    class_counts = defaultdict(int)
    selected = []

    for img_path in existing:
        label_path = label_path_from_image(img_path)
        instances = count_instances_in_file(label_path)

        can_add = True
        for class_id, count in instances.items():
            if class_counts[class_id] + count > max_instances:
                can_add = False
                break

        if can_add:
            selected.append(img_path)
            for class_id, count in instances.items():
                class_counts[class_id] += count

    return selected, class_counts


def copy_split(split_name, dataset_dir, output_dir, max_instances, seed):
    """Procesa un split (train/valid/test) y copia los archivos seleccionados."""
    split_dir    = os.path.normpath(os.path.join(dataset_dir, split_name))
    out_split_dir = os.path.normpath(os.path.join(output_dir, split_name))

    images_out = os.path.join(out_split_dir, "images")
    labels_out = os.path.join(out_split_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    image_paths = get_all_images(split_dir)
    if not image_paths:
        print(f"  [{split_name}] No se encontraron imágenes, saltando.")
        return

    print(f"\n  [{split_name}] Imágenes originales: {len(image_paths)}")

    if split_name == "train":
        selected, class_counts = select_images(image_paths, max_instances, seed)
        print(f"  [{split_name}] Imágenes seleccionadas: {len(selected)}")
        print(f"  [{split_name}] Instancias por clase tras balanceo:")
        for cls, cnt in sorted(class_counts.items()):
            print(f"             Clase {cls:3d}: {cnt} instancias")
    else:
        # valid y test: copiar solo las que existen
        selected = [p for p in image_paths if os.path.exists(p)]
        missing = len(image_paths) - len(selected)
        if missing > 0:
            print(f"    AVISO: {missing} imágenes no encontradas en disco, se ignoran.")
        print(f"  [{split_name}] Se copian todas las existentes: {len(selected)} imágenes.")

    for img_path in selected:
        img_name = os.path.basename(img_path)
        lbl_path = label_path_from_image(img_path)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"

        shutil.copy2(img_path, os.path.join(images_out, img_name))
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(labels_out, lbl_name))


def copy_yaml(dataset_dir, output_dir):
    """Copia el data.yaml al directorio de salida."""
    src = os.path.normpath(os.path.join(dataset_dir, "data.yaml"))
    dst = os.path.normpath(os.path.join(output_dir, "data.yaml"))
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"\n  data.yaml copiado a {dst}")
    else:
        print("\n  AVISO: no se encontró data.yaml en el dataset original.")


def main():
    print("=" * 60)
    print("  UNDERSAMPLING DE DATASET YOLO")
    print(f"  Límite: {MAX_INSTANCES} instancias por clase")
    print(f"  Seed:   {RANDOM_SEED}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in SPLITS:
        copy_split(split, DATASET_DIR, OUTPUT_DIR, MAX_INSTANCES, RANDOM_SEED)

    copy_yaml(DATASET_DIR, OUTPUT_DIR)

    print("\n  ¡Listo! Dataset balanceado guardado en:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()