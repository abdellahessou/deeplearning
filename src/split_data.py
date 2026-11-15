"""Étape 1 : refait le découpage train / val / test du dataset Kaggle.

Le dossier val d'origine ne contient que 16 images : on le fusionne avec le train,
puis on refait un split stratifié 85 % / 15 %. Le test d'origine reste intact.

Lancement :  python src/split_data.py
"""
import shutil

from sklearn.model_selection import train_test_split

from config import RAW_DIR, DATA_DIR, CLASSES, SEED, VAL_SIZE

# Extensions d'images acceptées (le dataset n'utilise que .jpeg, on reste prudent)
EXTENSIONS = {".jpeg", ".jpg", ".png"}


def lister_images(dossiers):
    """Retourne les chemins des images et leurs étiquettes (0 = NORMAL, 1 = PNEUMONIA)."""
    chemins, labels = [], []
    for d in dossiers:
        for label, classe in enumerate(CLASSES):
            # sorted() : ordre toujours identique, indispensable pour un split reproductible
            for p in sorted((RAW_DIR / d / classe).iterdir()):
                if p.suffix.lower() in EXTENSIONS:
                    chemins.append(p)
                    labels.append(label)
    return chemins, labels


def copier(chemins, labels, split):
    """Copie chaque image dans data/processed/<split>/<classe>/."""
    for p, y in zip(chemins, labels):
        dest = DATA_DIR / split / CLASSES[y]
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dest / p.name)


def main():
    if not (RAW_DIR / "train").exists():
        raise SystemExit(f"Dataset introuvable dans {RAW_DIR} : télécharge-le d'abord (voir README).")

    # On repart de zéro : sinon les images d'un ancien split resteraient mélangées au nouveau
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    # 1. Fusion du train et du val d'origine (le val n'a que 16 images)
    X, y = lister_images(["train", "val"])

    # 2. Split stratifié : même proportion NORMAL/PNEUMONIA dans train et val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=SEED)

    # 3. Le test d'origine reste intact
    X_te, y_te = lister_images(["test"])

    splits = {"train": (X_tr, y_tr), "val": (X_val, y_val), "test": (X_te, y_te)}
    for nom, (chemins, labels) in splits.items():
        copier(chemins, labels, nom)
        print(f"{nom:5s} : {len(chemins):4d} images | "
              f"NORMAL={labels.count(0):4d}  PNEUMONIA={labels.count(1):4d}")


if __name__ == "__main__":
    main()
