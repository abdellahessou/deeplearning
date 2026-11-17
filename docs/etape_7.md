# Étape 7 – Application Streamlit

## 1. Résumé

On a construit l'interface demandée par le sujet, `app/app.py`, avec **Streamlit** : une application web écrite en
Python pur, sans HTML ni JavaScript. L'utilisateur charge une radio (ou choisit un exemple du test), choisit le modèle
et le seuil de décision, puis voit côte à côte la radio et sa carte Grad-CAM, la prédiction et la probabilité de
pneumonie. L'application réutilise le code des étapes précédentes (`PREPROCESS`, Grad-CAM, lecture des images), pour
que l'image soit traitée **exactement** comme pendant l'entraînement. On l'a testée dans un navigateur : les
probabilités affichées sont identiques à celles de l'évaluation de l'étape 5.

## 2. Résultat obtenu

Commande : `streamlit run app/app.py` (ou `python -m streamlit run app/app.py`), puis http://localhost:8501.

### Ce que contient l'application

- **Barre latérale** :
  - liste des modèles présents dans `models/`, avec MobileNetV2 par défaut ;
  - curseur du seuil de décision, de 0,10 à 0,90 par pas de 0,05, à 0,50 par défaut ;
  - un rappel des performances de MobileNetV2 sur le test.
- **Page principale** :
  - un avertissement : l'outil ne remplace pas un diagnostic médical ;
  - une zone de chargement (JPEG, JPG, PNG) **ou** une liste de 6 radios d'exemple du test (3 normales, 3 pneumonies),
    pratique pour la démonstration en soutenance ;
  - la radio (224×224) et la radio avec la carte Grad-CAM, côte à côte ;
  - la prédiction, en vert (NORMAL) ou en rouge (PNEUMONIE), puis une barre de probabilité avec le seuil choisi ;
  - un message d'erreur clair si le fichier n'est pas une image lisible.

### Tests réalisés

| Test | Résultat |
|---|---|
| Lancement sans image | Message « Charge une radiographie… », aucune erreur |
| Modèles proposés | `mobilenetv2`, `resnet50`, `vgg19` |
| Radio d'exemple `person100_bacteria_475.jpeg`, MobileNetV2 | PNEUMONIE, 72,8 % (identique à l'étape 5) |
| Radio d'exemple `IM-0001-0001.jpeg` (normale), MobileNetV2 | NORMAL, 2,5 % (identique à l'étape 5) |
| Même radio, VGG19, seuil 0,10 | PNEUMONIE, 53,8 % (identique à l'étape 5) |
| Même radio, VGG19, seuil 0,60 | NORMAL : la prédiction et la carte Grad-CAM changent avec le seuil |
| Envoi d'un fichier PNG créé dans le navigateur | Image lue, prédiction affichée |
| Envoi d'un faux fichier `.png` (du texte) | Message « Impossible de lire ce fichier… », pas de plantage |
| Console du navigateur et journal du serveur | Aucune erreur |

Les tests ont été faits de deux façons : avec l'outil de test de Streamlit (`AppTest`, sans navigateur), puis à la main
dans un navigateur.

### Deux observations à présenter au jury

**1. Le modèle répond toujours, même à une image qui n'est pas une radio.** Pour tester l'envoi de fichier, on a créé
dans le navigateur une image synthétique (un simple dégradé gris). L'application l'a classée **PNEUMONIE à 100 %**.
Le modèle ne sait pas dire « je ne sais pas » ni « ce n'est pas une radio » : il a été entraîné à choisir entre deux
classes, et il en choisit toujours une. Un outil réel devrait d'abord vérifier que l'image est bien une radiographie
thoracique.

**2. Le biais « Clever Hans » est visible en direct.** Avec VGG19 sur la radio normale `IM-0001-0001.jpeg` (seuil 0,10,
prédiction PNEUMONIE), la carte Grad-CAM est rouge **sur la lettre « R »** en haut de l'image, comme on l'avait vu à
l'étape 6. Avec MobileNetV2, sur la même radio, la carte reste dans le thorax.

### Fichiers produits

- `app/app.py` : l'application (nouveau).
- `src/gradcam.py` : nouvelle fonction `decoder_image`, qui lit une image à partir de son contenu brut. `charger_image`
  l'utilise ; les figures de l'étape 6 régénérées sont identiques octet pour octet.
- `.streamlit/config.toml` : désactive l'envoi de statistiques d'utilisation à Streamlit.
- `.claude/launch.json` : configuration pour lancer l'application depuis Claude Code (facultatif).
- `README.md` : section « Application Streamlit ».

### Écarts par rapport au guide

- **Lecture de l'image avec `decoder_image`** (TensorFlow, interpolation bilinéaire) au lieu de PIL : le guide utilise
  `Image.resize` de PIL, qui ne redimensionne pas exactement comme `image_dataset_from_directory`. Les probabilités
  seraient légèrement différentes de celles du test. Avec notre version, elles sont identiques.
- **La carte Grad-CAM explique la classe prédite avec le seuil choisi**, au lieu de toujours expliquer « pneumonie ».
- **`compile=False`** au chargement du modèle (étape 4).
- **Radios d'exemple du test**, gestion des fichiers illisibles et message quand aucun modèle n'est présent : ajouts.
- **Prédiction en couleur** (`st.success` / `st.error`) au lieu d'un simple titre.

### Limites à connaître

- **Pas de détection des images hors sujet** (voir l'observation 1).
- **Premier chargement lent** : environ 15 secondes, le temps d'importer TensorFlow et de charger le modèle. Ensuite,
  une prédiction avec sa carte prend de l'ordre d'une demi-seconde avec MobileNetV2 (mesure de l'étape 6). Choisir
  un autre modèle pour la première fois demande de le charger : quelques secondes de plus.
- **Application locale** : elle tourne sur le PC et n'est pas en ligne. Les images chargées restent sur la machine.
- **Le rappel de performances de la barre latérale concerne MobileNetV2**, même si un autre modèle est choisi.

### À faire de ton côté

Prendre 3 ou 4 captures d'écran pour le rapport et la soutenance, par exemple :

1. une pneumonie bien détectée (MobileNetV2, `person100_bacteria_475.jpeg`) ;
2. une radio normale bien reconnue (MobileNetV2, `IM-0001-0001.jpeg`) ;
3. l'effet du seuil (VGG19, `IM-0001-0001.jpeg`, seuil 0,10 puis 0,60) ;
4. le Grad-CAM de VGG19 sur la lettre « R » (même radio, seuil 0,10).

## 3. Notions théoriques

**Inférence.** Utiliser un modèle déjà entraîné pour prédire sur une nouvelle donnée. Ici, une seule image à la fois.

**Cohérence entraînement / production.** L'image doit être préparée exactement comme pendant l'entraînement : mêmes
3 canaux, même taille, même méthode de redimensionnement, même `preprocess_input`. Sinon le modèle reçoit des
données qu'il n'a jamais vues et se trompe, sans aucun message d'erreur. C'est une erreur très fréquente en pratique.
On l'a évitée en réutilisant les fonctions des étapes précédentes, et vérifiée en comparant les probabilités.

**Streamlit.** Une bibliothèque qui transforme un script Python en page web. Chaque fois que l'utilisateur touche un
élément (curseur, liste, fichier), **tout le script est réexécuté** de haut en bas, avec les nouvelles valeurs.

**Mise en cache (`st.cache_resource`).** Comme le script est réexécuté à chaque clic, le modèle serait rechargé à
chaque fois (plusieurs secondes, jusqu'à 210 Mo pour ResNet50). `@st.cache_resource` garde le résultat de la fonction
en mémoire : le modèle n'est chargé qu'une fois par nom.

**Seuil de décision réglable.** Le modèle donne une probabilité ; le seuil la transforme en décision. Le laisser
réglable permet à l'utilisateur de choisir lui-même le compromis entre pneumonies ratées et fausses alertes (étape 5),
au lieu de figer un seuil choisi en regardant le test.

**Données hors distribution.** Une image très différente de celles de l'entraînement (une photo, un dessin, une radio
d'un autre organe). Un classifieur à deux classes répond quand même, souvent avec une grande confiance : une
probabilité élevée ne veut pas dire que le modèle « sait ».

## 4. Fonctions et classes

### Créées

| Nom | Fichier | Rôle | Paramètres |
|---|---|---|---|
| `charger_modele(nom)` | `app.py` | Charge un modèle `.keras`, mis en cache | nom du modèle |
| `exemples_test(n)` | `app.py` | Les n premières radios de chaque classe du test | `n` : 3 |
| `decoder_image(octets)` | `gradcam.py` | Décode une image brute comme le pipeline (RGB, 224×224, bilinéaire) | contenu du fichier |

### Utilisées

| Nom | Rôle | Paramètres importants |
|---|---|---|
| `st.set_page_config` | Titre de l'onglet et mise en page large | `page_title`, `layout="wide"` |
| `st.cache_resource` | Garde le modèle en mémoire | décorateur |
| `st.sidebar.selectbox`, `st.sidebar.slider` | Choix du modèle et du seuil | options ; min, max, défaut, pas |
| `st.file_uploader` | Zone de dépôt de fichier | `type=["jpeg", "jpg", "png"]` |
| `st.selectbox` | Liste des radios d'exemple | options |
| `st.columns`, `st.image` | Affichage côte à côte | nombre de colonnes ; `caption`, `width` |
| `st.success`, `st.error`, `st.warning`, `st.info` | Messages colorés | texte |
| `st.progress` | Barre de probabilité | valeur entre 0 et 1, `text` |
| `st.stop` | Arrête le script ici (rien d'autre à afficher) | aucun |
| `make_gradcam_heatmap`, `superposer` | Carte Grad-CAM et superposition (étape 6) | voir étape 6 |
| `PREPROCESS` | Prétraitement propre à chaque modèle (étape 2) | nom du modèle |

## 5. Le code important, expliqué ligne par ligne

### 5.1 Importer le code de `src/`

```python
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from config import MODELS_DIR, DATA_DIR, CLASSES
from data import PREPROCESS
from gradcam import make_gradcam_heatmap, superposer, decoder_image
```

- `Path(__file__).resolve()` : le chemin complet de `app.py`. `.parents[1]` remonte de deux niveaux
  (`app/` puis la racine du projet), on y ajoute `src`.
- `sys.path.append(...)` : ajoute ce dossier à la liste des endroits où Python cherche les modules. On peut alors
  importer `config`, `data` et `gradcam` comme dans les autres scripts.

### 5.2 Charger le modèle une seule fois

```python
@st.cache_resource
def charger_modele(nom):
    return keras.models.load_model(MODELS_DIR / f"{nom}.keras", compile=False)
```

- `@st.cache_resource` : au premier appel avec `"mobilenetv2"`, la fonction s'exécute et le modèle est gardé en
  mémoire ; aux appels suivants avec le même nom, Streamlit renvoie directement le modèle déjà chargé.
- `compile=False` : pour prédire, il suffit de l'architecture et des poids.

### 5.3 La barre latérale

```python
disponibles = [n for n in PREPROCESS if (MODELS_DIR / f"{n}.keras").exists()]
if not disponibles:
    st.error(...)
    st.stop()
nom = st.sidebar.selectbox("Modèle", disponibles, help=...)
seuil = st.sidebar.slider("Seuil de décision", 0.1, 0.9, 0.5, 0.05, help=...)
```

- `disponibles` : on ne propose que les modèles dont le fichier existe vraiment.
- `st.stop()` : s'il n'y en a aucun, on affiche une erreur et on arrête le script.
- `st.sidebar.selectbox` : une liste déroulante dans la barre latérale ; elle renvoie le nom choisi.
- `st.sidebar.slider("…", 0.1, 0.9, 0.5, 0.05)` : un curseur de 0,1 à 0,9, réglé sur 0,5, qui avance par pas de 0,05.
  `help` affiche une bulle d'aide.

### 5.4 Choisir la radio

```python
fichier = st.file_uploader("Charger une radiographie", type=["jpeg", "jpg", "png"])
choix = st.selectbox("… ou choisir une radio d'exemple du test set", ["(aucune)"] + list(exemples))
if fichier is not None:
    octets, nom_image = fichier.getvalue(), fichier.name
elif choix != "(aucune)":
    octets, nom_image = exemples[choix].read_bytes(), exemples[choix].name
else:
    st.info("Charge une radiographie ...")
    st.stop()
```

- `st.file_uploader` : renvoie `None` tant qu'aucun fichier n'est chargé, sinon un objet fichier.
- `fichier.getvalue()` : le contenu brut du fichier (des octets). `read_bytes()` fait la même chose pour un fichier
  d'exemple sur le disque. Les deux sources finissent donc au même format.
- Le fichier chargé est prioritaire sur l'exemple. Sans aucun des deux, on affiche une consigne et on s'arrête.

### 5.5 Préparer l'image et prédire

```python
try:
    img = decoder_image(octets)
except Exception:
    st.error("Impossible de lire ce fichier ...")
    st.stop()
x = PREPROCESS[nom](img[np.newaxis].copy())
model = charger_modele(nom)
proba = float(model.predict(x, verbose=0)[0, 0])
classe = int(proba >= seuil)
label = "PNEUMONIE" if classe == 1 else "NORMAL"
```

- `decoder_image(octets)` : la même lecture que pendant l'entraînement (RGB, 224×224, bilinéaire).
- `try / except` : si le fichier n'est pas une image valide, on affiche un message au lieu de planter.
- `img[np.newaxis]` : ajoute la dimension du lot, (224, 224, 3) devient (1, 224, 224, 3). `.copy()` protège l'image
  affichée, car le prétraitement peut modifier le tableau sur place.
- `PREPROCESS[nom]` : la normalisation propre au modèle choisi (étape 2).
- `model.predict(x)[0, 0]` : la probabilité de pneumonie de la première (et seule) image.
- `classe = int(proba >= seuil)` : la décision dépend du **seuil choisi** par l'utilisateur, pas de 0,5.

### 5.6 Grad-CAM et affichage

```python
heatmap = make_gradcam_heatmap(model, x, nom, classe)
img_uint8 = np.uint8(np.clip(np.round(img), 0, 255))
col1, col2 = st.columns(2)
col1.image(img_uint8, caption=..., width=420)
col2.image(superposer(img_uint8, heatmap), caption=..., width=420)
if classe == 1:
    st.error(f"Prédiction : **{label}**")
else:
    st.success(f"Prédiction : **{label}**")
st.progress(proba, text=f"Probabilité de pneumonie : {proba:.1%} (seuil : {seuil:.0%})")
```

- `make_gradcam_heatmap(..., classe)` : la carte explique la décision affichée.
- `np.clip(np.round(img), 0, 255)` puis `np.uint8` : l'image lue est en nombres décimaux ; on la remet en entiers de
  0 à 255 pour l'afficher et pour OpenCV.
- `st.columns(2)` : deux colonnes côte à côte ; on écrit dans chacune avec `col1.` et `col2.`.
- `st.error` / `st.success` : un bandeau rouge pour PNEUMONIE, vert pour NORMAL.
- `st.progress(proba, ...)` : une barre remplie à `proba` (entre 0 et 1). `{proba:.1%}` affiche 0,728 sous la forme
  « 72.8% ».

## 6. Trois questions de jury

**Q1. Comment garantissez-vous que l'application donne les mêmes résultats que votre évaluation ?**
L'image est préparée par les mêmes fonctions que pendant l'entraînement : `decoder_image` lit l'image en 3 canaux et
la redimensionne en 224×224 avec la même méthode (bilinéaire) que `image_dataset_from_directory`, puis `PREPROCESS`
applique la normalisation propre au modèle. Le code de référence du guide utilisait PIL pour redimensionner, avec une
méthode un peu différente. Nous l'avons vérifié : sur trois radios du test, l'application affiche exactement les mêmes
probabilités que l'évaluation de l'étape 5 (72,8 %, 2,5 % et 53,8 %).

**Q2. À quoi sert `st.cache_resource` ?**
Streamlit réexécute tout le script à chaque interaction : déplacer le curseur du seuil relance le code du début à la
fin. Sans cache, le modèle serait rechargé à chaque fois, ce qui prend plusieurs secondes (le fichier de ResNet50 pèse
210 Mo). Avec `@st.cache_resource`, il est chargé une seule fois, puis réutilisé. Le premier affichage prend environ
15 secondes ; ensuite, seuls la prédiction et le calcul de la carte sont refaits.

**Q3. Que se passe-t-il si on donne à l'application une image qui n'est pas une radiographie ?**
Le modèle répond quand même, et souvent avec assurance : une image synthétique en dégradé de gris a été classée
pneumonie à 100 %. C'est une limite importante. Le modèle n'a appris qu'à choisir entre NORMAL et PNEUMONIA sur des
radios d'enfants ; il n'a aucune notion de « ce n'est pas une radio ». Une probabilité de 100 % ne signifie donc pas
que le modèle est sûr d'avoir raison. Un outil réel devrait vérifier l'image avant de prédire, par exemple avec un
second modèle qui reconnaît les radiographies thoraciques. C'est aussi pour cela que l'application affiche un
avertissement.
