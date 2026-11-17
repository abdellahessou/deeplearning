"""Étape 6 : explicabilité avec Grad-CAM.

Produit dans reports/figures/ :
  gradcam_<modèle>.png     : 4 cas du test (vrai positif, vrai négatif, faux positif, faux négatif)
  gradcam_fp_<modèle>.png  : les 4 faux positifs les plus confiants

Lancement :  python src/gradcam.py --model mobilenetv2
"""
import argparse

import matplotlib
matplotlib.use("Agg")               # on enregistre les figures sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import MODELS_DIR, FIG_DIR, IMG_SIZE, CLASSES
from data import PREPROCESS
from models import LAST_CONV


def make_gradcam_heatmap(model, img_array, backbone, classe=1):
    """Carte Grad-CAM (valeurs entre 0 et 1) de la dernière couche convolutionnelle.

    img_array : image déjà prétraitée, de forme (1, 224, 224, 3).
    classe    : 1 = ce qui fait pencher vers PNEUMONIA, 0 = ce qui fait pencher vers NORMAL.
    """
    # 1. Un modèle qui renvoie à la fois les cartes de la dernière convolution
    #    et l'entrée de la couche de sortie (le vecteur de caractéristiques)
    conv_layer = model.get_layer(LAST_CONV[backbone])
    dense = model.layers[-1]                                    # Dense(1, sigmoid)
    grad_model = keras.Model(model.input, [conv_layer.output, dense.input])

    # 2. Gradient du score par rapport aux cartes de caractéristiques
    with tf.GradientTape() as tape:
        conv_out, features = grad_model(img_array, training=False)
        # Score AVANT la sigmoïde (logit) : quand la probabilité vaut 1,000, la sigmoïde
        # est « saturée » et son gradient devient nul ; le logit, lui, garde un gradient utile.
        logit = tf.matmul(features, dense.kernel) + dense.bias
        score = logit[:, 0] if classe == 1 else -logit[:, 0]
    grads = tape.gradient(score, conv_out)                      # même forme que conv_out

    # 3. Importance de chaque canal = moyenne de ses gradients sur toute la carte
    poids = tf.reduce_mean(grads, axis=(0, 1, 2))               # un nombre par canal

    # 4. Somme des cartes pondérées par leur importance, puis ReLU et normalisation entre 0 et 1
    heatmap = tf.reduce_sum(conv_out[0] * poids, axis=-1)       # 7x7 (14x14 pour VGG19)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def superposer(img_rgb, heatmap, alpha=0.4):
    """Superpose la carte en couleurs (bleu = peu important, rouge = très important) sur la radio.

    img_rgb : image en uint8, de forme (224, 224, 3).
    """
    h = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))      # 7x7 → 224x224 (lissé)
    h = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)          # valeurs → couleurs, en BGR
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)                              # OpenCV travaille en BGR
    return np.uint8(img_rgb * (1 - alpha) + h * alpha)


def charger_image(chemin):
    """Lit une radio exactement comme le pipeline de données : RGB, 224x224, interpolation bilinéaire."""
    img = tf.io.decode_image(tf.io.read_file(str(chemin)), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)                        # float32, valeurs 0..255
    return img.numpy()


def expliquer(model, backbone, chemin):
    """Renvoie (probabilité de PNEUMONIA, radio en uint8, radio + Grad-CAM de la classe prédite)."""
    img = charger_image(chemin)
    x = PREPROCESS[backbone](img[None].copy())                  # (1, 224, 224, 3), prétraitée
    prob = float(model.predict(x, verbose=0)[0, 0])
    classe = int(prob >= 0.5)                                   # on explique la décision prise
    heatmap = make_gradcam_heatmap(model, x, backbone, classe)
    img_uint8 = np.uint8(np.clip(np.round(img), 0, 255))
    return prob, img_uint8, superposer(img_uint8, heatmap)


def figure(model, backbone, cas, fichier, titre):
    """Une colonne par cas : la radio en haut, la radio + Grad-CAM en bas.

    cas : liste de (nom du cas, chemin de l'image, vraie classe 0/1).
    """
    fig, axes = plt.subplots(2, len(cas), figsize=(3.6 * len(cas), 8.5), layout="constrained")
    for col, (nom, chemin, vraie) in enumerate(cas):
        prob, img, superpose = expliquer(model, backbone, chemin)
        predite = CLASSES[int(prob >= 0.5)]
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{nom}\nvérité : {CLASSES[vraie]}\nP(pneumonie) = {prob:.3f}", fontsize=10)
        axes[1, col].imshow(superpose)
        axes[1, col].set_title(f"Grad-CAM de la prédiction : {predite}", fontsize=10)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle(titre, fontsize=13)
    fig.savefig(FIG_DIR / fichier, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Figure :", FIG_DIR / fichier)


if __name__ == "__main__":
    from evaluate import predire        # réutilise la prédiction sur le test de l'étape 5

    p = argparse.ArgumentParser(description="Cartes Grad-CAM sur des radios du test set.")
    p.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "resnet50", "vgg19"])
    name = p.parse_args().model
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    y_true, y_prob, chemins = predire(name)
    model = keras.models.load_model(MODELS_DIR / f"{name}.keras", compile=False)

    # Pour chaque type de cas, l'exemple où le modèle est le plus sûr de lui
    malade, sain = y_true == 1, y_true == 0
    def plus_haut(masque): return np.where(masque)[0][np.argmax(y_prob[masque])]
    def plus_bas(masque): return np.where(masque)[0][np.argmin(y_prob[masque])]
    cas = [("Vrai positif", plus_haut(malade)),
           ("Vrai négatif", plus_bas(sain)),
           ("Faux positif", plus_haut(sain)),
           ("Faux négatif", plus_bas(malade & (y_prob < 0.5)))]
    figure(model, name, [(nom, chemins[i], y_true[i]) for nom, i in cas],
           f"gradcam_{name}.png", f"Grad-CAM – {name} (cas les plus confiants du test)")

    # Les 4 fausses alertes les plus confiantes (étape 5) : sur quoi le modèle s'appuie-t-il ?
    fp = np.where(sain & (y_prob >= 0.5))[0]
    fp = fp[np.argsort(-y_prob[fp])][:4]
    figure(model, name, [(f"Faux positif n° {k + 1}", chemins[i], 0) for k, i in enumerate(fp)],
           f"gradcam_fp_{name}.png", f"Grad-CAM – {name} : les 4 faux positifs les plus confiants")
