"""Étape 4 : compare les modèles entraînés sur le test set.

Produit reports/comparaison.csv et reports/figures/comparaison_modeles.png.
Lancement :  python src/compare.py
"""
import time

import matplotlib
matplotlib.use("Agg")               # on enregistre la figure sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow import keras

from config import MODELS_DIR, FIG_DIR, BATCH_SIZE
from data import PREPROCESS, get_datasets


def evaluer(name, seuil=0.5):
    """Évalue un modèle sur le test set et renvoie ses métriques dans un dictionnaire."""
    _, _, test = get_datasets(name)
    # compile=False : on ne recharge que l'architecture et les poids, ce qui suffit pour prédire.
    # Sinon le modèle garde les réglages d'entraînement de Colab (jit_compile=True, prévu pour le GPU),
    # qui rendent les prédictions 4 fois plus lentes sur le processeur du PC.
    model = keras.models.load_model(MODELS_DIR / f"{name}.keras", compile=False)

    # On charge tout le test en mémoire (624 images) : images déjà normalisées + vraies étiquettes
    x = np.concatenate([images.numpy() for images, _ in test])
    y_true = np.concatenate([labels.numpy() for _, labels in test]).ravel().astype(int)

    model.predict(x[:BATCH_SIZE], verbose=0)                    # tour de chauffe, non chronométré
    t0 = time.perf_counter()
    y_prob = model.predict(x, batch_size=BATCH_SIZE, verbose=0).ravel()    # probabilités de PNEUMONIA
    ms_image = (time.perf_counter() - t0) / len(y_true) * 1000

    y_pred = (y_prob >= seuil).astype(int)                      # probabilité → classe
    return {"modèle": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob),
            "paramètres (M)": model.count_params() / 1e6,
            "ms / image": ms_image}


def tracer_comparaison(df):
    """Diagramme en barres : F1 et recall de chaque modèle."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    largeur = 0.38
    for i, (metrique, couleur) in enumerate([("f1", "#2a78d6"), ("recall", "#eb6834")]):
        barres = ax.bar(x + (i - 0.5) * (largeur + 0.02), df[metrique], largeur,
                        label=metrique, color=couleur)
        ax.bar_label(barres, fmt="%.3f", padding=3)             # valeur écrite au-dessus de la barre
    ax.set_xticks(x, df["modèle"])
    ax.set_ylim(0, 1.2)                                         # place pour la légende au-dessus des barres
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score sur le test set")
    ax.set_title("F1 et recall des modèles (test set, seuil 0,5)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper center")
    fig.savefig(FIG_DIR / "comparaison_modeles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # On ne compare que les modèles réellement présents dans models/
    noms = [n for n in PREPROCESS if (MODELS_DIR / f"{n}.keras").exists()]
    if not noms:
        raise SystemExit(f"Aucun modèle .keras dans {MODELS_DIR} : lance d'abord l'entraînement.")

    df = pd.DataFrame([evaluer(n) for n in noms]).round(3)
    df.to_csv(FIG_DIR.parent / "comparaison.csv", index=False)
    tracer_comparaison(df)
    print(df.to_string(index=False))
    print("Tableau :", FIG_DIR.parent / "comparaison.csv")
    print("Figure  :", FIG_DIR / "comparaison_modeles.png")
