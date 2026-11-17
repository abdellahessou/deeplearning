# Détection de pneumonie sur radiographies thoraciques

Projet académique de Deep Learning (5IIR) : classification binaire **NORMAL vs PNEUMONIA**
à partir de radiographies thoraciques, avec des CNN pré-entraînés (MobileNetV2, ResNet50, VGG19),
une explication visuelle des prédictions (Grad-CAM) et une application Streamlit.

> Outil pédagogique : il ne remplace pas un diagnostic médical.

## Installation

```bash
py -3.11 -m venv .venv311       # Linux / macOS : python3.11 -m venv .venv311
.venv311\Scripts\activate       # Linux / macOS : source .venv311/bin/activate
pip install -r requirements.txt
```

Vérifier que TensorFlow fonctionne et s'il voit un GPU :

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

Une liste vide `[]` signifie que la machine travaille sur CPU : l'entraînement se fait alors sur Google Colab.

## Données

1. Créer un jeton sur kaggle.com > Settings > API, puis l'enregistrer dans `~/.kaggle/access_token`
   (ou `kaggle.json` pour une clé « legacy »). Ne jamais le mettre dans le projet.
2. Télécharger le dataset, puis refaire le split train / val / test :

```bash
python -m kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw --unzip
python src/split_data.py
```

L'archive contient un dossier en double (`chest_xray/chest_xray`) et un dossier `__MACOSX` : on ne garde que
`data/raw/chest_xray/` avec ses sous-dossiers `train`, `val` et `test`.

L'exploration (répartition des classes, exemples de radios) est dans `notebooks/01_exploration.ipynb`.

Tester le pipeline de données (9 images augmentées dans `reports/figures/augmentation.png` et poids des classes) :

```bash
python src/data.py
```

## Entraînement

```bash
python src/train.py --model mobilenetv2 --epochs2 0    # phase 1 seule (base gelée)
python src/train.py --model resnet50                   # phase 1 + fine-tuning
```

Options : `--model` (`mobilenetv2`, `resnet50`, `vgg19`), `--epochs1` (15 par défaut), `--epochs2` (10 par défaut, 0 = pas de fine-tuning).

Le script produit dans `models/` le modèle final (`<modèle>.keras`), le meilleur modèle de chaque phase et
les historiques (`<modèle>_phase1_history.csv`, `<modèle>_phase2_history.csv`), et dans `reports/figures/` les courbes
(`courbes_<modèle>.png`).

Sans GPU, utiliser `notebooks/colab_train.ipynb` sur Google Colab (Exécution > Modifier le type d'exécution > GPU T4).
Ce notebook est prévu pour Colab uniquement : ne pas l'exécuter sur le PC.

## Comparaison des modèles

```bash
python src/compare.py
```

Évalue sur le test set tous les modèles présents dans `models/` et produit `reports/comparaison.csv`
(accuracy, precision, recall, F1, AUC, nombre de paramètres, temps d'inférence par image) ainsi que
`reports/figures/comparaison_modeles.png`.

## Évaluation détaillée

```bash
python src/evaluate.py --model mobilenetv2
```

Évalue un modèle sur le test set et produit :

- `reports/rapport_<modèle>.txt` : precision, recall et F1 par classe ;
- `reports/seuils_<modèle>.csv` : recall, precision, faux négatifs et faux positifs pour les seuils 0,3 à 0,7 ;
- dans `reports/figures/` : matrice de confusion (`cm_`), courbe ROC (`roc_`), courbe précision-rappel (`pr_`),
  et les erreurs les plus confiantes (`erreurs_fn_`, `erreurs_fp_`).

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
- [x] Étape 1 – Données : téléchargement et nouveau split
- [x] Étape 2 – Pipeline de données, augmentation, poids des classes
- [x] Étape 3 – Modèle de base et entraînement (transfer learning)
- [x] Étape 4 – Fine-tuning et comparaison des 3 modèles
- [x] Étape 5 – Évaluation complète
- [ ] Étape 6 – Explicabilité avec Grad-CAM
- [ ] Étape 7 – Application Streamlit
- [ ] Étape 8 – Rapport, README et soutenance
