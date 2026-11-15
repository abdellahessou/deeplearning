# Étape 2 – Pipeline de données, augmentation et poids des classes

## 1. Résumé

On a construit dans `src/data.py` le « tuyau » qui amène les images du disque jusqu'au modèle. Il lit les radios de
`data/processed/`, les redimensionne en 224×224, les regroupe par lots de 32, puis les normalise exactement comme
l'attend chaque modèle pré-entraîné (MobileNetV2, ResNet50, VGG19). Pour le train uniquement, il applique en plus une
**augmentation** : de petites transformations aléatoires qui évitent au modèle d'apprendre les images par cœur.
Enfin, on calcule des **poids de classes** pour compenser le déséquilibre (74 % de pneumonies).
Aucun modèle n'est encore entraîné : cette étape prépare seulement ce qu'il va manger.

## 2. Résultat obtenu

Sortie exacte du test intégré (`python src/data.py`) :

```
Figure sauvegardée : ...\reports\figures\augmentation.png
Class weights : {0: 1.94, 1: 0.67}
mobilenetv2 | lot (32, 224, 224, 3) | pixels de -1.0 à 1.0 | étiquettes (32, 1)
resnet50    | lot (32, 224, 224, 3) | pixels de -123.7 à 151.1 | étiquettes (32, 1)
vgg19       | lot (32, 224, 224, 3) | pixels de -123.7 à 151.1 | étiquettes (32, 1)
```

- **Poids des classes** : `{0: 1.94, 1: 0.67}`, exactement la valeur annoncée par le guide.
  NORMAL pèse 2,88 fois plus que PNEUMONIA, ce qui correspond au ratio des effectifs (3 300 / 1 147).
- **Figure** : `reports/figures/augmentation.png`, 9 radios du train après augmentation. Elles restent réalistes.

### Vérifications effectuées

| Vérification | Résultat |
|---|---|
| Effectifs lus : train / val / test | 4 447 / 785 / 624 (identiques à l'étape 1) |
| Test **non mélangé** | même ordre à deux passages : 234 NORMAL puis 390 PNEUMONIA |
| Validation **sans augmentation** | pixels strictement identiques à deux passages |
| Train **remélangé** à chaque epoch | ordre différent entre deux passages |
| Augmentation réellement aléatoire | deux appels sur le même lot donnent des résultats différents ; 8 copies d'une même image donnent 8 images différentes |
| Augmentation coupée hors entraînement | avec `training=False`, l'image ressort inchangée |
| Pixels après augmentation | restent entre 0 et 255 |
| Images grises → 3 canaux | les 3 canaux sont identiques avant normalisation |
| Normalisation MobileNetV2 | valeurs entre -1 et 1 |
| Normalisation ResNet50 / VGG19 | valeurs entre -123,68 et 151,06, canaux décalés différemment (moyennes ImageNet) |
| Reproductibilité | deux lancements du script produisent exactement la même figure |

### Fichiers produits

- `src/data.py` : le pipeline complet.
- `reports/figures/augmentation.png` : les 9 images augmentées.

### Limites à connaître

- **Images déformées** : les radios d'origine ne sont pas carrées (médiane 1272×888). Le redimensionnement en 224×224
  les étire un peu, sans garder les proportions. C'est le comportement standard et il est identique pour train, val et test.
- **Bords en miroir** : quand une rotation ou une translation découvre un coin vide, Keras le remplit par défaut avec
  le reflet de l'image (`fill_mode="reflect"`). On voit parfois un petit motif en miroir dans un coin. L'effet est
  discret et touche les deux classes de la même façon.
- **Lots déséquilibrés** : le mélange est aléatoire, donc un lot peut contenir beaucoup plus de pneumonies qu'un autre
  (de 66 % à 94 % sur les 5 premiers lots observés). Les poids de classes compensent en moyenne.

### Écart par rapport au guide

- Les couches d'augmentation reçoivent `seed=SEED`, pour respecter la règle « seed 42 partout » de `CLAUDE.md`.
- Le chemin du dossier est converti en texte (`str(...)`) avant d'être donné à Keras, par sécurité.
- `num_parallel_calls=AUTOTUNE` est aussi appliqué à la normalisation de val et test (simple gain de vitesse).

## 3. Notions théoriques

**Pipeline de données (`tf.data`).** Une chaîne d'opérations appliquées aux images à la volée : lire, redimensionner,
regrouper en lots, augmenter, normaliser. Les 4 447 images ne sont jamais toutes chargées en mémoire : le pipeline
fournit un lot de 32 à la fois, comme un tapis roulant.

**Lot (batch).** Le modèle ne voit pas les images une par une, mais par paquets de 32. Un lot est un tableau de forme
`(32, 224, 224, 3)` : 32 images, 224 pixels de haut, 224 de large, 3 canaux de couleur.

**Data augmentation.** À chaque epoch, chaque image du train est légèrement modifiée au hasard (un peu tournée,
zoomée, décalée, plus ou moins contrastée). Le modèle ne voit donc jamais deux fois exactement la même image. Avec
seulement 4 447 radios, cela réduit le **surapprentissage** : il doit apprendre ce qu'est une pneumonie, pas retenir
des images. On ne l'applique **jamais** à la validation ni au test, qui doivent représenter de vraies radios.

**Justification de chaque augmentation.** Une transformation est acceptable si le résultat ressemble encore à une
radio qu'un hôpital pourrait produire, et si le diagnostic ne change pas.

| Augmentation | Réglage | Pourquoi c'est réaliste |
|---|---|---|
| Rotation | ±10° | Un patient (ici un enfant) n'est jamais parfaitement droit devant le détecteur |
| Zoom | ±10 % | La distance au détecteur et la taille du patient varient |
| Translation | ±5 % | Le patient n'est pas toujours bien centré |
| Contraste | ±10 % | Les réglages d'exposition changent d'un appareil ou d'un cliché à l'autre |
| Flip vertical | **non** | Une radio n'est jamais prise la tête en bas : cela créerait des images impossibles |
| Flip horizontal | **non** | Le cœur passerait à droite, ce qui est une anomalie anatomique rare |

Les amplitudes restent faibles exprès : une rotation de 45° ou un zoom de 50 % donnerait des images qu'on ne
rencontre jamais, ou couperait les poumons.

**Normalisation (`preprocess_input`).** Chaque modèle a été entraîné sur ImageNet avec une mise à l'échelle précise
des pixels. Il faut lui donner exactement la même, sinon ses filtres ne reconnaissent plus rien.
MobileNetV2 attend des valeurs entre -1 et 1. ResNet50 et VGG19 attendent des canaux dans l'ordre BGR (bleu, vert,
rouge), auxquels on a retiré la moyenne d'ImageNet. C'est pour cela que `get_datasets` prend le nom du modèle en paramètre.

**Gris → 3 canaux.** Une radio a un seul canal (niveaux de gris). Les modèles ImageNet attendent 3 canaux (couleur).
On recopie donc le canal gris trois fois.

**Poids de classes (class weights).** Dans la loss, chaque erreur est multipliée par le poids de sa classe.
La formule est `total / (2 × effectif de la classe)` : NORMAL reçoit 4 447 / (2 × 1 147) = 1,94 et PNEUMONIA
4 447 / (2 × 3 300) = 0,67. Une erreur sur une radio normale compte donc environ 3 fois plus. Sans cela, le modèle
aurait intérêt à répondre « pneumonie » trop souvent. Avec ces poids, les deux classes pèsent exactement pareil au
total (2 223,5 chacune).

**Prefetch.** Pendant que le modèle calcule sur un lot, le processeur prépare déjà le suivant. Le modèle n'attend
jamais ses données.

## 4. Fonctions et classes

### Créées dans `src/data.py`

| Nom | Rôle | Paramètres |
|---|---|---|
| `PREPROCESS` | Dictionnaire : nom du modèle → sa fonction de normalisation | clés `"mobilenetv2"`, `"resnet50"`, `"vgg19"` |
| `augmentation` | Petit modèle Keras qui enchaîne les 4 transformations aléatoires | un lot d'images, `training=True` pour l'activer |
| `charger_split(split, shuffle)` | Lit un dossier de `data/processed/` en lots d'images étiquetées | `split` : `"train"`, `"val"` ou `"test"` ; `shuffle` : mélanger ou non |
| `get_datasets(backbone)` | Renvoie les 3 jeux (train, val, test) prêts pour un modèle | `backbone` : nom du modèle |
| `class_weights()` | Calcule le poids de chaque classe à partir du train | aucun |

### Utilisées

| Nom | Rôle | Paramètres importants |
|---|---|---|
| `keras.utils.image_dataset_from_directory` | Charge les images de dossiers en `tf.data.Dataset` | `label_mode`, `class_names`, `color_mode`, `image_size`, `batch_size`, `shuffle`, `seed` |
| `layers.RandomRotation` | Rotation aléatoire | fraction de tour : `10 / 360` = ±10° |
| `layers.RandomZoom` | Zoom aléatoire | `0.1` = ±10 % |
| `layers.RandomTranslation` | Décalage aléatoire | `0.05, 0.05` = ±5 % en hauteur et en largeur |
| `layers.RandomContrast` | Contraste aléatoire | `0.1` = ±10 % |
| `keras.Sequential` | Enchaîne des couches dans l'ordre | liste de couches |
| `Dataset.map` | Applique une fonction à chaque lot | la fonction, `num_parallel_calls` |
| `Dataset.prefetch` | Prépare les lots suivants en avance | `tf.data.AUTOTUNE` |
| `preprocess_input` | Normalisation propre à chaque modèle | un lot d'images (pixels 0 à 255) |

## 5. Le code important, expliqué ligne par ligne

### 5.1 Le dictionnaire des normalisations

```python
PREPROCESS = {
    "mobilenetv2": keras.applications.mobilenet_v2.preprocess_input,
    "resnet50": keras.applications.resnet50.preprocess_input,
    "vgg19": keras.applications.vgg19.preprocess_input,
}
```

- Un dictionnaire associe une clé (le nom du modèle) à une valeur (ici une **fonction**, sans parenthèses : on ne
  l'appelle pas, on la range).
- `PREPROCESS["resnet50"]` renvoie donc la fonction de normalisation de ResNet50.
- Avantage : le reste du code est le même pour les 3 modèles, seul le nom change. Ce dictionnaire resservira dans
  l'application Streamlit, pour normaliser l'image exactement comme à l'entraînement.

### 5.2 L'augmentation

```python
augmentation = keras.Sequential([
    layers.RandomRotation(10 / 360, seed=SEED),
    layers.RandomZoom(0.1, seed=SEED),
    layers.RandomTranslation(0.05, 0.05, seed=SEED),
    layers.RandomContrast(0.1, seed=SEED),
], name="augmentation")
```

- `keras.Sequential([...])` : les 4 couches sont appliquées l'une après l'autre.
- `RandomRotation(10 / 360)` : Keras exprime l'angle en fraction de tour complet. 10/360 de tour = 10 degrés.
  L'angle est tiré au hasard entre -10° et +10°.
- `RandomZoom(0.1)` : zoom avant ou arrière, jusqu'à 10 %.
- `RandomTranslation(0.05, 0.05)` : décalage jusqu'à 5 % de la hauteur et 5 % de la largeur.
- `RandomContrast(0.1)` : contraste augmenté ou diminué, jusqu'à 10 %.
- `seed=SEED` : les tirages sont reproductibles d'un lancement à l'autre, tout en restant différents d'une image à l'autre.
- Il n'y a **pas** de `RandomFlip` : c'est un choix volontaire (voir le tableau des justifications).

### 5.3 Lire un split

```python
def charger_split(split, shuffle):
    return keras.utils.image_dataset_from_directory(
        str(DATA_DIR / split),
        labels="inferred",
        label_mode="binary",
        class_names=CLASSES,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED)
```

- `str(DATA_DIR / split)` : le dossier à lire, par exemple `data/processed/train`.
- `labels="inferred"` : l'étiquette est déduite du nom du sous-dossier (`NORMAL` ou `PNEUMONIA`). C'est pour cela
  qu'on a rangé les images ainsi à l'étape 1.
- `label_mode="binary"` : les étiquettes sont 0 ou 1, ce qui convient à une sortie sigmoïde.
- `class_names=CLASSES` : on impose l'ordre, 0 = NORMAL et 1 = PNEUMONIA. Le modèle prédira donc « la probabilité de pneumonie ».
- `color_mode="rgb"` : chaque radio grise est convertie en 3 canaux identiques.
- `image_size=IMG_SIZE` : tout est redimensionné en 224×224.
- `batch_size=BATCH_SIZE` : les images sortent par lots de 32.
- `shuffle=shuffle, seed=SEED` : mélange ou non, de façon reproductible.

### 5.4 Assembler les trois jeux

```python
def get_datasets(backbone):
    prep = PREPROCESS[backbone]
    AUTOTUNE = tf.data.AUTOTUNE

    train = charger_split("train", shuffle=True)
    val = charger_split("val", shuffle=False)
    test = charger_split("test", shuffle=False)

    train = train.map(lambda x, y: (augmentation(x, training=True), y),
                      num_parallel_calls=AUTOTUNE)
    train = train.map(lambda x, y: (prep(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val = val.map(lambda x, y: (prep(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test = test.map(lambda x, y: (prep(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train, val, test
```

- `prep = PREPROCESS[backbone]` : on récupère la normalisation du modèle demandé.
- `AUTOTUNE` : TensorFlow choisit lui-même combien de calculs faire en parallèle.
- `shuffle=True` pour le train : le modèle voit les images dans un ordre différent à chaque epoch, ce qui l'empêche
  d'apprendre l'ordre. `shuffle=False` pour val et test : l'ordre reste fixe.
  Pour le test c'est **indispensable** : à l'étape 5, on comparera les prédictions aux vraies étiquettes, il faut donc
  que les deux soient dans le même ordre.
- Premier `map` sur le train : `lambda x, y: (...)` est une petite fonction sans nom. `x` est un lot d'images, `y`
  ses étiquettes. On augmente `x` et on laisse `y` intact. `training=True` active les transformations aléatoires.
- Deuxième `map` : la normalisation. **L'ordre compte** : on augmente d'abord (sur des pixels de 0 à 255), on
  normalise ensuite.
- `val` et `test` : seulement la normalisation, **aucune augmentation**.
- `.prefetch(AUTOTUNE)` : le lot suivant est préparé pendant que le modèle travaille.

### 5.5 Les poids de classes

```python
def class_weights():
    n = {c: len(list((DATA_DIR / "train" / c).glob("*"))) for c in CLASSES}
    total = sum(n.values())
    return {i: total / (2 * n[c]) for i, c in enumerate(CLASSES)}
```

- `n` : un dictionnaire avec le nombre d'images de chaque classe dans le train, soit `{"NORMAL": 1147, "PNEUMONIA": 3300}`.
  On compte **uniquement le train** : c'est le seul jeu sur lequel la loss est pondérée.
- `total` : 4 447.
- La dernière ligne applique la formule `total / (2 × effectif)` et renvoie `{0: 1.94, 1: 0.67}`. Les clés sont
  0 et 1 (et non les noms), car c'est le format attendu par `model.fit(class_weight=...)` à l'étape 3.
- Le `2` est le nombre de classes : avec des classes équilibrées, la formule donnerait 1 partout.

## 6. Trois questions de jury

**Q1. Pourquoi n'appliquez-vous l'augmentation que sur le train, et pas de flip ?**
L'augmentation sert à faire varier ce que le modèle voit pendant l'apprentissage, pour qu'il généralise mieux.
La validation et le test servent à mesurer : ils doivent contenir de vraies radios, non modifiées, sinon on mesurerait
la performance sur des images artificielles et le score changerait à chaque passage. Nous avons vérifié que la
validation renvoie exactement les mêmes pixels à deux passages. Pour les flips : une radio n'est jamais prise à
l'envers, et un flip horizontal placerait le cœur à droite. On ne garde que des transformations médicalement plausibles.

**Q2. Pourquoi la normalisation dépend-elle du modèle ? Que se passerait-il si on se trompait ?**
Chaque modèle a appris ses filtres sur ImageNet avec une mise à l'échelle précise : entre -1 et 1 pour MobileNetV2,
canaux BGR centrés sur la moyenne d'ImageNet pour ResNet50 et VGG19. Si on donne à un modèle des valeurs d'une autre
échelle, ses filtres reçoivent des nombres qu'ils n'ont jamais vus et les caractéristiques extraites n'ont plus de
sens. Le piège est qu'il n'y a aucun message d'erreur : le modèle s'entraîne quand même, mais mal. C'est pour cela que
la même fonction, rangée dans `PREPROCESS`, est réutilisée à l'entraînement, à l'évaluation et dans l'application.

**Q3. Comment avez-vous géré le déséquilibre des classes, et pourquoi des poids plutôt que du sur-échantillonnage ?**
Avec des poids de classes : chaque erreur est multipliée par 1,94 pour une radio normale et 0,67 pour une pneumonie,
selon la formule `total / (2 × effectif)`. Les deux classes pèsent alors autant dans la loss. Par rapport au
sur-échantillonnage (recopier des radios normales), c'est plus simple, sans image dupliquée et sans risque
d'apprendre par cœur les copies. Les poids ne règlent que l'entraînement : pour l'évaluation, on utilise en plus des
métriques adaptées (recall, F1, AUC) et on pourra ajuster le seuil de décision à l'étape 5.
