"""Étape 2 : pipeline de données (lecture, augmentation, normalisation) et poids des classes.

Test rapide :  python src/data.py
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import DATA_DIR, FIG_DIR, IMG_SIZE, BATCH_SIZE, SEED, CLASSES

# Chaque modèle pré-entraîné attend ses propres valeurs de pixels
PREPROCESS = {
    "mobilenetv2": keras.applications.mobilenet_v2.preprocess_input,  # → valeurs dans [-1, 1]
    "resnet50": keras.applications.resnet50.preprocess_input,         # → BGR, centré sur ImageNet
    "vgg19": keras.applications.vgg19.preprocess_input,               # → BGR, centré sur ImageNet
}

# Transformations aléatoires réalistes pour une radio (pas de flip vertical ni horizontal)
augmentation = keras.Sequential([
    layers.RandomRotation(10 / 360, seed=SEED),          # ±10 degrés : patient légèrement penché
    layers.RandomZoom(0.1, seed=SEED),                   # ±10 % : distance au détecteur variable
    layers.RandomTranslation(0.05, 0.05, seed=SEED),     # ±5 % : patient pas parfaitement centré
    layers.RandomContrast(0.1, seed=SEED),               # ±10 % : réglages d'exposition différents
], name="augmentation")


def charger_split(split, shuffle):
    """Lit data/processed/<split>/ : un sous-dossier par classe = une étiquette."""
    return keras.utils.image_dataset_from_directory(
        str(DATA_DIR / split),
        labels="inferred",          # l'étiquette est déduite du nom du sous-dossier
        label_mode="binary",        # étiquettes 0 ou 1 (sortie sigmoïde)
        class_names=CLASSES,        # impose l'ordre : 0 = NORMAL, 1 = PNEUMONIA
        color_mode="rgb",           # gris → 3 canaux, comme l'attendent les modèles ImageNet
        image_size=IMG_SIZE,        # redimensionne tout en 224x224
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED)


def get_datasets(backbone):
    """Renvoie (train, val, test) prêts pour le modèle `backbone`."""
    prep = PREPROCESS[backbone]
    AUTOTUNE = tf.data.AUTOTUNE     # TensorFlow choisit lui-même le niveau de parallélisme

    train = charger_split("train", shuffle=True)
    val = charger_split("val", shuffle=False)
    test = charger_split("test", shuffle=False)     # ordre fixe pour l'évaluation

    # Augmentation UNIQUEMENT sur le train, puis normalisation partout
    train = train.map(lambda x, y: (augmentation(x, training=True), y),
                      num_parallel_calls=AUTOTUNE)
    train = train.map(lambda x, y: (prep(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val = val.map(lambda x, y: (prep(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test = test.map(lambda x, y: (prep(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train, val, test


def class_weights():
    """Poids inversement proportionnels à la taille de chaque classe (calculés sur le train)."""
    n = {c: len(list((DATA_DIR / "train" / c).glob("*"))) for c in CLASSES}
    total = sum(n.values())
    return {i: total / (2 * n[c]) for i, c in enumerate(CLASSES)}


if __name__ == "__main__":
    # Petit test : 9 images augmentées + poids des classes + un lot du pipeline complet
    import matplotlib.pyplot as plt

    # 1. Neuf images du train, augmentées (avant normalisation, pour pouvoir les afficher)
    images, labels = next(iter(charger_split("train", shuffle=True)))
    aug = augmentation(images[:9], training=True)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, img, y in zip(axes.flat, aug, labels):
        ax.imshow(img.numpy().astype("uint8"))
        ax.set_title(CLASSES[int(y[0])], fontsize=10)
        ax.axis("off")
    fig.suptitle("9 images du train après augmentation")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "augmentation.png", dpi=150, bbox_inches="tight")
    print("Figure sauvegardée :", FIG_DIR / "augmentation.png")

    # 2. Poids des classes
    print("Class weights :", {i: round(w, 2) for i, w in class_weights().items()})

    # 3. Un lot du pipeline complet pour chaque modèle
    for nom in PREPROCESS:
        train, _, _ = get_datasets(nom)
        x, y = next(iter(train))
        print(f"{nom:11s} | lot {x.shape} | pixels de {float(tf.reduce_min(x)):.1f} "
              f"à {float(tf.reduce_max(x)):.1f} | étiquettes {y.shape}")
