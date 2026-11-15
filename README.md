# Détection de pneumonie sur radiographies thoraciques

Projet académique de Deep Learning (5IIR) : classification binaire **NORMAL vs PNEUMONIA**
à partir de radiographies thoraciques, avec des CNN pré-entraînés (MobileNetV2, ResNet50, VGG19),
une explication visuelle des prédictions (Grad-CAM) et une application Streamlit.

> Outil pédagogique : il ne remplace pas un diagnostic médical.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt
```

Vérifier que TensorFlow fonctionne et s'il voit un GPU :

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

Une liste vide `[]` signifie que la machine travaille sur CPU : l'entraînement se fait alors sur Google Colab.

## Structure du projet

```
├── CLAUDE.md           # instructions permanentes pour Claude Code
├── requirements.txt    # dépendances Python
├── data/               # dataset (non versionné)
│   ├── raw/chest_xray/ # dataset Kaggle d'origine
│   └── processed/      # train / val / test refaits
├── src/                # code du projet (config.py = constantes)
├── app/                # application Streamlit
├── notebooks/          # exploration + notebook Colab
├── models/             # modèles .keras sauvegardés (non versionnés)
├── reports/figures/    # figures du rapport
└── docs/               # résumé expliqué de chaque étape
```

## Avancement

- [x] Étape 0 – Environnement et structure
- [ ] Étape 1 – Données : téléchargement et nouveau split
- [ ] Étape 2 – Pipeline de données, augmentation, poids des classes
- [ ] Étape 3 – Modèle de base et entraînement (transfer learning)
- [ ] Étape 4 – Fine-tuning et comparaison des 3 modèles
- [ ] Étape 5 – Évaluation complète
- [ ] Étape 6 – Explicabilité avec Grad-CAM
- [ ] Étape 7 – Application Streamlit
- [ ] Étape 8 – Rapport, README et soutenance
