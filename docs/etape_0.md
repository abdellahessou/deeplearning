# Étape 0 – Environnement et structure

## 1. Résumé

On a préparé le « chantier » avant d'écrire la moindre ligne de Deep Learning : les dossiers du projet,
un environnement virtuel Python isolé, les bibliothèques (TensorFlow, scikit-learn, Streamlit…), un dépôt Git
et un fichier central `src/config.py` qui contient toutes les constantes (taille d'image, seed, chemins).
Le but est double : que le projet soit **reproductible** (n'importe qui peut le réinstaller avec `requirements.txt`)
et **cohérent** (tous les scripts lisent les mêmes valeurs au même endroit, donc pas de 224 ici et 256 ailleurs).
On a aussi vérifié que TensorFlow fonctionne et constaté que ce PC n'a pas de GPU utilisable : l'entraînement se fera sur Google Colab.

## 2. Résultat obtenu

| Élément | Résultat |
|---|---|
| Système | Windows 11, processeur AMD Ryzen 7, 20 Go de RAM |
| Python de l'environnement virtuel | **3.11.9** (dans `.venv311/`) |
| TensorFlow | **2.21.0** (avec Keras 3.15.1) – s'importe sans erreur |
| GPU détecté par TensorFlow | `[]` → **aucun**, calcul sur CPU |
| Dépôt Git | initialisé (branche `main`), premier commit effectué |

Sortie exacte de la commande de vérification :

```
> python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
2.21.0 []
```

Tests effectués (tous réussis) :

| Test | Résultat |
|---|---|
| Import de `src/config.py` | `ROOT` = `C:\Projects\deeplearning`, `(224, 224) 32 42 ['NORMAL', 'PNEUMONIA'] 0.15` |
| Import des autres bibliothèques | numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, Pillow 13.0.0, OpenCV 5.0.0, matplotlib 3.11.2, seaborn 0.13.2, streamlit 1.67.0 |
| Mini-modèle Keras (1 neurone sigmoïde) sur 2 exemples | sortie de forme `(2, 1)` : Keras calcule bien sur CPU |

Remarque : juste après une installation, le **tout premier** import de TensorFlow peut être très lent
(plusieurs minutes : fichiers fraîchement installés, probablement analysés par l'antivirus). Les imports
suivants prennent une dizaine de secondes. Si la commande semble bloquée, il suffit de la relancer.

Fichiers et dossiers créés :

```
deeplearning/
├── CLAUDE.md            # instructions permanentes pour Claude Code
├── README.md            # présentation minimale + installation
├── requirements.txt     # liste des bibliothèques
├── .gitignore           # ce que Git doit ignorer
├── .venv311/            # environnement virtuel Python 3.11 (ignoré par Git)
├── data/raw/chest_xray/ # recevra le dataset Kaggle (ignoré par Git)
├── data/processed/      # recevra nos splits train/val/test (ignoré par Git)
├── src/config.py        # constantes du projet
├── app/                 # application Streamlit (étape 7)
├── notebooks/           # exploration + notebook Colab
├── models/              # modèles .keras (ignorés par Git)
├── reports/figures/     # figures du rapport
└── docs/etape_0.md      # ce fichier
```

**Écarts par rapport au guide (à connaître pour la soutenance) :**

- **Python 3.11 installé à côté du Python du PC.** Le PC n'avait que Python 3.14, trop récent : TensorFlow n'y existe
  qu'en pré-version (`2.22.0rc0`). On a donc installé Python 3.11.9, la version recommandée par le guide, sans
  remplacer le 3.14. L'environnement s'appelle `.venv311` pour rappeler la version de Python qu'il contient.
- **Pas de GPU en local.** La carte graphique est une AMD Radeon intégrée. TensorFlow n'accélère les calculs
  qu'avec des cartes NVIDIA (technologie CUDA), et de toute façon plus du tout sous Windows natif depuis la version 2.11.
- **`.gitignore` un peu plus complet** que demandé : on a ajouté `kaggle.json` (clé secrète) et `.ipynb_checkpoints/`.
- **Fichiers `.gitkeep`** dans les dossiers vides : Git n'enregistre pas un dossier vide, ce petit fichier vide le force à le garder.
- Le dossier s'appelle `deeplearning` et non `pneumonia-xray` : aucun impact, les chemins sont calculés automatiquement.

## 3. Notions théoriques

**Environnement virtuel (`.venv311`).** C'est une copie isolée de Python, propre au projet. Les bibliothèques installées
dedans ne se mélangent pas avec celles des autres projets. Image simple : chaque projet a sa propre boîte à outils,
au lieu d'une seule boîte commune où les outils de versions différentes se marchent dessus.

**`requirements.txt`.** La liste des bibliothèques nécessaires. Avec `pip install -r requirements.txt`, n'importe qui
(le jury, Colab, un coéquipier) recrée le même environnement en une commande.

**GPU vs CPU.** Entraîner un CNN, c'est faire des millions de multiplications de matrices. Un CPU a quelques cœurs
très polyvalents ; un GPU a des milliers de petits cœurs qui font ces multiplications en parallèle. Résultat :
l'entraînement est 10 à 50 fois plus rapide sur GPU. Sans GPU, on peut quand même écrire et tester le code en local,
puis lancer le vrai entraînement sur Colab (GPU T4 gratuit).

**Reproductibilité et seed.** Beaucoup d'opérations sont aléatoires : le découpage train/validation, le mélange des images,
l'initialisation des poids. Une *seed* (graine) fixe, ici 42, fait que ces tirages « aléatoires » donnent exactement
le même résultat à chaque exécution. Sans elle, on ne pourrait pas comparer deux modèles honnêtement.

**Fichier de configuration central.** Toutes les constantes sont dans un seul fichier. Si on veut changer la taille
des images, on modifie une ligne, et tous les scripts suivent. Cela évite les incohérences difficiles à repérer.

**Git et `.gitignore`.** Git garde l'historique du code (chaque *commit* est une photo du projet). Le `.gitignore`
liste ce qu'on ne veut pas versionner : les données (plus d'1 Go), les modèles (dizaines de Mo), l'environnement
virtuel (recréable) et les secrets (clé Kaggle).

## 4. Fonctions, classes et commandes utilisées

| Nom | Rôle | Paramètres |
|---|---|---|
| `pathlib.Path` | Représente un chemin de fichier, compatible Windows / Linux / Colab | le chemin en texte |
| `Path(__file__)` | Chemin du fichier Python en cours d'exécution | `__file__` : variable créée automatiquement par Python |
| `.resolve()` | Transforme le chemin en chemin absolu complet | aucun |
| `.parents[n]` | Remonte de n+1 dossiers (`parents[0]` = dossier du fichier, `parents[1]` = celui au-dessus) | l'indice `n` |
| Opérateur `/` sur un `Path` | Colle un sous-dossier au chemin (`ROOT / "models"`) | le nom à ajouter |
| `py -3.11 -m venv .venv311` | Crée l'environnement virtuel dans le dossier `.venv311` | `-3.11` : version de Python ; nom du dossier |
| `pip install -r requirements.txt` | Installe toutes les bibliothèques listées | `-r` : lire la liste dans un fichier |
| `tf.config.list_physical_devices('GPU')` | Liste les GPU que TensorFlow peut utiliser (`[]` = aucun) | le type d'appareil |
| `git init` | Crée un dépôt Git vide dans le dossier | `-b main` : nom de la branche |

Aucune fonction ni classe n'a été créée à cette étape : `config.py` ne contient que des constantes.

## 5. Le code important, expliqué ligne par ligne

### 5.1 `src/config.py` – les chemins

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "chest_xray"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
```

- `from pathlib import Path` : on importe l'outil standard de Python pour manipuler les chemins.
- `ROOT = ...` : la ligne la plus importante. `__file__` vaut `…/deeplearning/src/config.py`. `.resolve()` en fait un
  chemin absolu. `.parents[0]` serait `…/deeplearning/src`, donc `.parents[1]` est `…/deeplearning` : **la racine du projet**.
  Avantage : le chemin est calculé à partir de l'emplacement du fichier, pas de l'endroit d'où on lance le script.
  Le même code marche donc sur mon PC et sur Colab, sans rien modifier.
- `RAW_DIR` : là où sera le dataset Kaggle tel qu'on le télécharge (on n'y touche jamais).
- `DATA_DIR` : là où on rangera nos propres splits train / val / test (étape 1).
- `MODELS_DIR` : là où seront sauvegardés les modèles entraînés (`.keras`).
- `FIG_DIR` : là où iront toutes les figures du rapport.

### 5.2 `src/config.py` – les paramètres

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASSES = ["NORMAL", "PNEUMONIA"]
VAL_SIZE = 0.15
```

- `IMG_SIZE = (224, 224)` : toutes les radios seront redimensionnées en 224×224 pixels. C'est la taille avec laquelle
  MobileNetV2, ResNet50 et VGG19 ont été entraînés sur ImageNet.
- `BATCH_SIZE = 32` : le modèle voit les images par paquets (*batches*) de 32. Il corrige ses poids après chaque paquet.
  32 est un compromis classique entre vitesse, mémoire et stabilité.
- `SEED = 42` : la graine aléatoire utilisée partout (split, mélange, initialisation). Le nombre lui-même n'a rien
  de spécial, l'important est qu'il soit **fixe**.
- `CLASSES = ["NORMAL", "PNEUMONIA"]` : l'ordre compte. La position dans la liste donne l'étiquette :
  0 = NORMAL, 1 = PNEUMONIA. La sortie du modèle sera donc « la probabilité de pneumonie ».
- `VAL_SIZE = 0.15` : 15 % des images d'entraînement seront mises de côté pour la validation (étape 1).

### 5.3 `.gitignore`

```
data/
models/*.keras
__pycache__/
.ipynb_checkpoints/
.venv311/
kaggle.json
```

- `data/` : tout le dossier des données est ignoré (trop lourd, et retéléchargeable depuis Kaggle).
- `models/*.keras` : on ignore seulement les fichiers `.keras` du dossier `models/`. Les petits fichiers CSV
  d'historique d'entraînement, eux, seront versionnés.
- `__pycache__/` et `.ipynb_checkpoints/` : fichiers temporaires créés par Python et Jupyter.
- `.venv311/` : l'environnement virtuel (des centaines de Mo, recréable avec `requirements.txt`).
- `kaggle.json` : la clé secrète de l'API Kaggle. La publier sur GitHub permettrait à n'importe qui d'utiliser ton compte.

### 5.4 Les commandes de mise en place

```bash
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
git init -b main
```

- Ligne 1 : `py -3.11` choisit Python 3.11 (et non le 3.14 par défaut) ; `-m venv .venv311` crée l'environnement virtuel.
- Ligne 2 : *active* l'environnement. Après cela, `python` et `pip` désignent ceux de `.venv311`. À refaire à chaque nouveau terminal.
- Ligne 3 : installe toutes les bibliothèques de `requirements.txt` dans l'environnement.
- Ligne 4 : importe TensorFlow, affiche sa version et la liste des GPU. Chez nous : `2.21.0 []`.
- Ligne 5 : crée le dépôt Git avec une branche nommée `main`.

## 6. Trois questions de jury

**Q1. Pourquoi utiliser un environnement virtuel et un `requirements.txt` ?**
Pour isoler le projet et le rendre reproductible. Les bibliothèques de Deep Learning évoluent vite et sont exigeantes
sur les versions : on l'a vu concrètement, TensorFlow n'a pas de version stable pour Python 3.14. Avec un environnement
dédié en Python 3.11 et la liste des dépendances, n'importe qui peut réinstaller le projet à l'identique, en local ou sur Colab.

**Q2. Votre machine n'a pas de GPU : comment avez-vous entraîné les modèles, et pourquoi le GPU est-il important ?**
En local, TensorFlow renvoie une liste de GPU vide : la carte est une AMD intégrée, alors que TensorFlow s'appuie sur
CUDA, réservé aux cartes NVIDIA. Le code est donc écrit et testé en local sur CPU, et l'entraînement est lancé sur
Google Colab avec un GPU T4. Le GPU est important car un CNN repose sur des multiplications de matrices massivement
parallélisables : l'entraînement y est 10 à 50 fois plus rapide.

**Q3. À quoi sert la seed fixée à 42, et pourquoi centraliser les constantes dans `config.py` ?**
La seed rend les tirages aléatoires identiques d'une exécution à l'autre (split, mélange, initialisation des poids) :
les résultats sont reproductibles et les trois modèles sont comparés dans les mêmes conditions. Centraliser les
constantes garantit que tous les scripts utilisent les mêmes valeurs : si l'entraînement utilisait des images 224×224
et l'application des images 256×256, les prédictions seraient fausses sans aucun message d'erreur.
