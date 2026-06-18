"""
balance_dataset.py
------------------
Undersampling inteligente de un dataset YOLO (estructura Roboflow).
Reduce las clases mayoritarias limitando sus instancias a un máximo configurable,
pero PROTEGE las imágenes que contienen clases minoritarias/raras para evitar
que se pierdan muestras críticas y causen falsos negativos (background).

"""

import os
import random
import shutil
from collections import defaultdict

# =============================================================================
# CONFIG
# =============================================================================

DATASET_DIR = "C:/Users/menci/TFGMencia/ModificacionesDataset/DatasetTFG-2502.v2i.yolov11"
OUTPUT_DIR  = "C:/Users/menci/TFGMencia/ModificacionesDataset/datasetBalanceado-v3-2"
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
    Selecciona imágenes priorizando las clases minoritarias (más raras)
    para evitar que desaparezcan debido al descarte de clases mayoritarias.
    
    Estrategia:
      1. Escanea globalmente el split para mapear qué clases son escasas (< MAX_INSTANCES).
      2. Separa las imágenes en 'valiosas' (tienen clases raras) y 'comunes'.
      3. Añade TODAS las valiosas sin restricción para salvaguardar la cola del dataset.
      4. Añade las comunes de forma aleatoria hasta rellenar el cupo de las mayoritarias.
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

    print("    [Analizando densidad de clases para protección de minorías]...")
    global_counts = defaultdict(int)
    image_contents = {}
    
    # Mapeo previo del contenido de cada imagen
    for img_path in existing:
        label_path = label_path_from_image(img_path)
        instances = count_instances_in_file(label_path)
        image_contents[img_path] = instances
        for class_id, count in instances.items():
            global_counts[class_id] += count

    valiosas = []
    comunes = []
    
    # Separar imágenes valiosas (contienen clases que globalmente no llegan al máximo)
    for img_path in existing:
        instances = image_contents[img_path]
        tiene_clase_rara = any(global_counts[cid] <= max_instances for cid in instances.keys())
        
        if tiene_clase_rara:
            valiosas.append(img_path)
        else:
            comunes.append(img_path)

    # Mezclar ambos grupos por separado para mantener aleatoriedad
    random.shuffle(valiosas)
    random.shuffle(comunes)

    class_counts = defaultdict(int)
    selected = []

    # PASO 1: Proteger y añadir imágenes de clases minoritarias
    for img_path in valiosas:
        selected.append(img_path)
        for class_id, count in image_contents[img_path].items():
            class_counts[class_id] += count

    # PASO 2: Rellenar con imágenes comunes respetando el límite superior
    for img_path in comunes:
        instances = image_contents[img_path]
        
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
        print(f"  [{split_name}] Instancias por clase tras balanceo inteligente:")
        for cls, cnt in sorted(class_counts.items()):
            print(f"             Clase {cls:3d}: {cnt} instancias")
    else:
        # valid y test: copiar solo las que existen sin alterar proporciones
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
    print("  UNDERSAMPLING INTELIGENTE (PROTECCIÓN DE CLASES RARAS)")
    print(f"  Límite Clases Altas: {MAX_INSTANCES} instancias")
    print(f"  Seed:                {RANDOM_SEED}")
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