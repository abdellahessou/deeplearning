"""Étape 7 : application Streamlit de détection de pneumonie.

Lancement (depuis la racine du projet) :  streamlit run app/app.py
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow import keras

# Permet d'importer les modules de src/ (config, data, gradcam)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from config import MODELS_DIR, DATA_DIR, CLASSES
from data import PREPROCESS
from gradcam import make_gradcam_heatmap, superposer, decoder_image

st.set_page_config(page_title="Détection de pneumonie", layout="wide")


@st.cache_resource                  # le modèle n'est chargé qu'une seule fois, puis gardé en mémoire
def charger_modele(nom):
    # compile=False : pour prédire, l'architecture et les poids suffisent (voir étape 4)
    return keras.models.load_model(MODELS_DIR / f"{nom}.keras", compile=False)


def exemples_test(n=3):
    """Les n premières radios de chaque classe du test set, pour une démonstration rapide."""
    exemples = {}
    for classe in CLASSES:
        dossier = DATA_DIR / "test" / classe
        if dossier.exists():
            for chemin in sorted(dossier.iterdir())[:n]:
                exemples[f"{classe} – {chemin.name}"] = chemin
    return exemples


st.title("Détection de pneumonie sur radiographie thoracique")
st.warning("Outil expérimental à visée pédagogique : il ne remplace pas un diagnostic médical. "
           "Seul un médecin peut interpréter une radiographie.")

# --- Barre latérale : choix du modèle et du seuil ---
disponibles = [n for n in PREPROCESS if (MODELS_DIR / f"{n}.keras").exists()]
if not disponibles:
    st.error(f"Aucun modèle trouvé dans {MODELS_DIR}. Lance d'abord l'entraînement (src/train.py).")
    st.stop()
nom = st.sidebar.selectbox("Modèle", disponibles,
                           help="MobileNetV2 est le modèle retenu : aussi bon que les autres, et le plus rapide.")
seuil = st.sidebar.slider("Seuil de décision", 0.1, 0.9, 0.5, 0.05,
                          help="Au-dessus de ce seuil, la radio est classée PNEUMONIE. "
                               "Plus bas : moins de pneumonies ratées, mais plus de fausses alertes.")
st.sidebar.caption("Sur le test set (624 radios, seuil 0,5), MobileNetV2 détecte 98,7 % des pneumonies, "
                   "mais classe 40 % des radios normales en pneumonie.")

# --- Page principale : choix de la radio ---
fichier = st.file_uploader("Charger une radiographie", type=["jpeg", "jpg", "png"])
exemples = exemples_test()
choix = st.selectbox("… ou choisir une radio d'exemple du test set", ["(aucune)"] + list(exemples))

if fichier is not None:
    octets, nom_image = fichier.getvalue(), fichier.name
elif choix != "(aucune)":
    octets, nom_image = exemples[choix].read_bytes(), exemples[choix].name
else:
    st.info("Charge une radiographie (JPEG ou PNG) ou choisis un exemple pour obtenir une prédiction.")
    st.stop()

# --- Prédiction ---
try:
    img = decoder_image(octets)             # même lecture que pendant l'entraînement : RGB, 224x224
except Exception:
    st.error("Impossible de lire ce fichier : vérifie qu'il s'agit bien d'une image JPEG ou PNG.")
    st.stop()

x = PREPROCESS[nom](img[np.newaxis].copy())     # (1, 224, 224, 3), prétraitée pour ce modèle
model = charger_modele(nom)
proba = float(model.predict(x, verbose=0)[0, 0])          # probabilité de PNEUMONIA
classe = int(proba >= seuil)                               # 1 = PNEUMONIE, 0 = NORMAL
label = "PNEUMONIE" if classe == 1 else "NORMAL"

# Grad-CAM de la classe prédite : ce qui a fait pencher le modèle vers sa réponse
heatmap = make_gradcam_heatmap(model, x, nom, classe)
img_uint8 = np.uint8(np.clip(np.round(img), 0, 255))

# --- Affichage ---
col1, col2 = st.columns(2)
col1.image(img_uint8, caption=f"Radiographie ({nom_image}, redimensionnée en 224×224)", width=420)
col2.image(superposer(img_uint8, heatmap), caption="Zones déterminantes (Grad-CAM) : rouge = très important",
           width=420)

if classe == 1:
    st.error(f"Prédiction : **{label}**")
else:
    st.success(f"Prédiction : **{label}**")
st.progress(proba, text=f"Probabilité de pneumonie : {proba:.1%} (seuil : {seuil:.0%})")
st.caption(f"Modèle : {nom}. La carte Grad-CAM montre où le modèle a regardé, pas une lésion confirmée : "
           "sa résolution est grossière (7×7 ou 14×14 cases).")
