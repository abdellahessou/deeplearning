"""Étape 5 : évaluation détaillée d'un modèle sur le test set.

Produit dans reports/ : le rapport de classification et le tableau des seuils,
et dans reports/figures/ : matrice de confusion, courbe ROC, courbe précision-rappel,
grilles des faux négatifs et des faux positifs les plus confiants.

Lancement :  python src/evaluate.py --model mobilenetv2
"""
import argparse

import matplotlib
matplotlib.use("Agg")               # on enregistre les figures sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from tensorflow import keras

from config import MODELS_DIR, FIG_DIR, CLASSES
from data import get_datasets, charger_split

BLEU, ORANGE = "#2a78d6", "#eb6834"
REPORTS_DIR = FIG_DIR.parent        # reports/


def predire(name):
    """Renvoie les vraies étiquettes, les probabilités de PNEUMONIA et le chemin de chaque image du test."""
    _, _, test = get_datasets(name)
    # compile=False : seuls l'architecture et les poids servent à prédire (voir étape 4)
    model = keras.models.load_model(MODELS_DIR / f"{name}.keras", compile=False)
    y_true = np.concatenate([y.numpy() for _, y in test]).ravel().astype(int)
    y_prob = model.predict(test, verbose=0).ravel()
    # Le test n'est pas mélangé : les fichiers sont dans le même ordre que les prédictions
    chemins = np.array(charger_split("test", shuffle=False).file_paths)
    return y_true, y_prob, chemins


def matrice_confusion(y_true, y_pred, name):
    """Heatmap des 4 cas : TN, FP / FN, TP, avec le pourcentage de chaque classe réelle."""
    cm = confusion_matrix(y_true, y_pred)                       # [[TN, FP], [FN, TP]]
    pct = cm / cm.sum(axis=1, keepdims=True) * 100              # % de chaque ligne (classe réelle)
    textes = [[f"{cm[i, j]}\n({pct[i, j]:.1f} %)" for j in range(2)] for i in range(2)]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=textes, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"fontsize": 13})
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(f"Matrice de confusion – {name} (seuil 0,5)")
    fig.savefig(FIG_DIR / f"cm_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def courbe_roc(y_true, y_prob, name, seuil=0.5):
    """Courbe ROC : taux de vrais positifs (recall) en fonction du taux de faux positifs, pour tous les seuils."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)                     # un point par seuil possible
    y_pred = y_prob >= seuil
    point = (y_pred[y_true == 0].mean(), y_pred[y_true == 1].mean())   # (FPR, TPR) au seuil choisi

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=BLEU, linewidth=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Hasard (AUC = 0,5)")
    ax.plot(*point, "o", color=ORANGE, markersize=9, label=f"Seuil {seuil:g}".replace(".", ","))
    ax.set_xlabel("Taux de faux positifs (1 – spécificité)")
    ax.set_ylabel("Taux de vrais positifs (recall)")
    ax.set_title(f"Courbe ROC – {name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG_DIR / f"roc_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def courbe_precision_rappel(y_true, y_prob, name, seuil=0.5):
    """Courbe précision-rappel : la précision obtenue pour chaque niveau de recall."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)                # aire sous cette courbe
    y_pred = y_prob >= seuil
    point = (y_pred[y_true == 1].mean(), y_true[y_pred].mean())  # (recall, precision) au seuil choisi
    base = y_true.mean()                                         # précision d'un modèle qui dit toujours PNEUMONIA

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, color=BLEU, linewidth=2, label=f"{name} (AP = {ap:.3f})")
    ax.axhline(base, linestyle="--", color="gray", label=f"Hasard (= part de pneumonies, {base:.3f})")
    ax.plot(*point, "o", color=ORANGE, markersize=9, label=f"Seuil {seuil:g}".replace(".", ","))
    ax.set_xlabel("Recall (pneumonies détectées)")
    ax.set_ylabel("Precision (alertes justes)")
    ax.set_title(f"Courbe précision-rappel – {name}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIG_DIR / f"pr_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ap


def tableau_seuils(y_true, y_prob, seuils=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Pour chaque seuil : recall, precision, spécificité, nombre de faux négatifs et de faux positifs."""
    lignes = []
    for s in seuils:
        y_pred = (y_prob >= s).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        lignes.append({"seuil": s,
                       "recall": tp / (tp + fn),
                       "precision": tp / (tp + fp),
                       "spécificité": tn / (tn + fp),
                       "accuracy": (tp + tn) / len(y_true),
                       "FN": fn, "FP": fp})
    return pd.DataFrame(lignes).round(3)


def grille_erreurs(y_true, y_prob, chemins, name, type_erreur, n=8, seuil=0.5):
    """Affiche les n erreurs dont le modèle était le plus sûr.

    type_erreur = "fn" : vraies pneumonies avec la probabilité la plus basse (le modèle était sûr que c'était normal).
    type_erreur = "fp" : radios normales avec la probabilité la plus haute (le modèle était sûr que c'était malade).
    """
    if type_erreur == "fn":
        idx = np.where((y_true == 1) & (y_prob < seuil))[0]
        idx = idx[np.argsort(y_prob[idx])][:n]                  # probabilités croissantes
        titre = "Faux négatifs : pneumonies classées NORMAL"
    else:
        idx = np.where((y_true == 0) & (y_prob >= seuil))[0]
        idx = idx[np.argsort(-y_prob[idx])][:n]                 # probabilités décroissantes
        titre = "Faux positifs : radios normales classées PNEUMONIA"
    if len(idx) == 0:
        print(f"Aucune erreur de type {type_erreur.upper()} : pas de grille.")
        return idx

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, idx):
        img = Image.open(chemins[i]).convert("L")               # image d'origine, en niveaux de gris
        ax.imshow(img, cmap="gray")
        nom_fichier = chemins[i].replace("\\", "/").split("/")[-1]
        ax.set_title(f"{nom_fichier}\nP(pneumonie) = {y_prob[i]:.3f}", fontsize=9)
    fig.suptitle(f"{titre} – {name} ({len(idx)} les plus confiants)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"erreurs_{type_erreur}_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return idx


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Évalue un modèle entraîné sur le test set.")
    p.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "resnet50", "vgg19"])
    name = p.parse_args().model
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    y_true, y_prob, chemins = predire(name)
    y_pred = (y_prob >= 0.5).astype(int)
    # Sécurité : l'étiquette lue dans le dossier de chaque fichier doit être celle du dataset
    assert all(CLASSES[t] in c.replace("\\", "/").split("/")[-2] for t, c in zip(y_true, chemins))

    # 1. Rapport de classification
    rapport = classification_report(y_true, y_pred, target_names=CLASSES, digits=3)
    print(f"=== {name} – test set ({len(y_true)} images), seuil 0,5 ===")
    print(rapport)
    (REPORTS_DIR / f"rapport_{name}.txt").write_text(rapport, encoding="utf-8")

    # 2. Matrice de confusion
    (tn, fp), (fn, tp) = matrice_confusion(y_true, y_pred, name)
    print(f"Matrice de confusion : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"Spécificité = {tn / (tn + fp):.3f}")

    # 3. Courbes ROC et précision-rappel
    courbe_roc(y_true, y_prob, name)
    ap = courbe_precision_rappel(y_true, y_prob, name)
    print(f"AUC ROC = {auc(*roc_curve(y_true, y_prob)[:2]):.3f} | AP (aire précision-rappel) = {ap:.3f}")

    # 4. Effet du seuil de décision
    seuils = tableau_seuils(y_true, y_prob)
    seuils.to_csv(REPORTS_DIR / f"seuils_{name}.csv", index=False)
    print("\nEffet du seuil :")
    print(seuils.to_string(index=False))

    # 5. Les erreurs les plus confiantes
    for type_erreur in ("fn", "fp"):
        idx = grille_erreurs(y_true, y_prob, chemins, name, type_erreur)
        print(f"\n{type_erreur.upper()} les plus confiants :")
        for i in idx:
            print(f"  {chemins[i].replace(chr(92), '/').split('/')[-1]:32s} P(pneumonie) = {y_prob[i]:.3f}")
    print("\nFigures dans", FIG_DIR)
