# Étape 3 – Modèle de base et entraînement (transfer learning)

## 1. Résumé

On a construit et entraîné le premier modèle du projet. Plutôt que d'apprendre un CNN à partir de zéro avec seulement
4 447 radios, on réutilise **MobileNetV2**, déjà entraîné sur 1,2 million d'images (ImageNet), comme extracteur de
caractéristiques. Ses couches sont **gelées** : on n'entraîne qu'une petite « tête » de 1 281 paramètres qui répond
NORMAL ou PNEUMONIA. `src/models.py` construit le modèle (le même code sert pour ResNet50 et VGG19) et `src/train.py`
l'entraîne avec les poids de classes et quatre callbacks qui surveillent l'entraînement. Un notebook Colab est prêt
pour les entraînements plus lourds de l'étape 4.

## 2. Résultat obtenu

### Entraînement de MobileNetV2, phase 1 (base gelée)

Commande : `python src/train.py --model mobilenetv2 --epochs2 0`, lancée sur le processeur du PC (environ 58 minutes pour 15 epochs).

| Epoch | Loss train | Accuracy train | Loss val | Accuracy val | Recall val | AUC val | Learning rate |
|---|---|---|---|---|---|---|---|
| 1 | 0,392 | 81,3 % | 0,182 | 94,3 % | 99,0 % | 0,983 | 0,001 |
| 3 | 0,175 | 92,7 % | 0,132 | 95,3 % | 97,6 % | 0,986 | 0,001 |
| 6 | 0,150 | 93,9 % | 0,108 | 96,6 % | 98,5 % | 0,990 | 0,001 |
| 9 | 0,136 | 94,4 % | 0,107 | 95,5 % | 95,5 % | 0,991 | 0,0002 |
| **11** | 0,123 | 95,2 % | **0,105** | **95,5 %** | 95,5 % | 0,991 | 0,0002 |
| 15 | 0,136 | 94,5 % | 0,107 | 95,5 % | 95,4 % | 0,991 | 0,00004 |

**Modèle retenu : celui de l'epoch 11** (plus petite loss de validation). Rechargé depuis `models/mobilenetv2.keras`
et réévalué sur la validation, il donne :

| Métrique (validation, 785 images) | Valeur |
|---|---|
| Loss | 0,105 |
| Accuracy | 95,5 % |
| Precision | 98,4 % |
| Recall | 95,5 % |
| AUC | 0,991 |

C'est dans la fourchette annoncée par le guide (93 à 96 % d'accuracy en validation).

**Le test set n'a pas été utilisé** : conformément à la règle du projet, les métriques finales ne seront calculées
sur le test qu'à l'étape 5. Les chiffres ci-dessus servent seulement à piloter l'entraînement.

### Comment lire les courbes (`reports/figures/courbes_mobilenetv2.png`)

- **La loss baisse vite puis se stabilise** : l'essentiel est appris en 5 à 6 epochs, ensuite les gains sont minimes.
- **Pas de surapprentissage** : la loss de validation ne remonte pas, elle reste plate autour de 0,105 à 0,11.
- **La validation est meilleure que le train**, ce qui surprend au premier regard. Trois raisons, toutes normales :
  le train est mesuré avec le dropout actif et sur des images augmentées (plus difficiles) ; la loss du train est
  multipliée par les poids de classes, pas celle de validation ; et la validation contient des patients déjà vus
  dans le train (limite notée à l'étape 1).
- **Le learning rate a été réduit deux fois** par `ReduceLROnPlateau` : de 0,001 à 0,0002 après l'epoch 8, puis à
  0,00004 après l'epoch 13.
- **L'arrêt anticipé ne s'est pas déclenché** : il faut 5 epochs sans amélioration, et il n'y en a eu que 4 après
  l'epoch 11. Les 15 epochs ont été faites, puis les poids de l'epoch 11 ont été restaurés.

### Les trois architectures (vérifiées par un mini-entraînement de test)

| Modèle | Paramètres au total | Entraînables en phase 1 | Dernière couche conv (Grad-CAM) | Sortie de cette couche |
|---|---|---|---|---|
| MobileNetV2 | 2 259 265 | 1 281 | `out_relu` | 7×7×1280 |
| ResNet50 | 23 589 761 | 2 049 | `conv5_block3_out` | 7×7×2048 |
| VGG19 | 20 024 897 | 513 | `block5_conv4` | 14×14×512 |

Pour chacun, sur quelques lots : phase 1, fine-tuning, sauvegarde, rechargement et prédictions identiques après
rechargement ont été vérifiés. Ce test valide le code, pas la performance : seul MobileNetV2 a été réellement entraîné.

### Fichiers produits

- `src/models.py`, `src/train.py` : le code.
- `notebooks/colab_train.ipynb` : notebook pour Colab. **Non testé sur Colab** (je ne peux pas l'y exécuter) ;
  ses commandes sont les mêmes que celles testées en local.
- `models/mobilenetv2.keras` (9,4 Mo) : le modèle final, non versionné.
- `models/mobilenetv2_phase1.keras` : le meilleur modèle de la phase, non versionné.
- `models/mobilenetv2_phase1_history.csv` : l'historique epoch par epoch.
- `reports/figures/courbes_mobilenetv2.png` : les courbes loss et accuracy.

### Écarts par rapport au guide

- **Entraînement fait en local sur CPU** et non sur Colab : MobileNetV2 gelé est assez léger pour cela. Les trois
  modèles de l'étape 4 (avec fine-tuning) devront passer par Colab.
- `build_model` reçoit aussi `input_shape` : sans cela, Keras affiche un avertissement trompeur pour MobileNetV2.
- `train.py` trace les courbes à la fin, et efface les anciens historiques avant de commencer.
- Sur Colab, le jeton Kaggle est demandé à l'écran (`getpass`) au lieu d'un fichier `kaggle.json` à uploader.

### Point à traiter à l'étape 4

`fine_tune(base, 30)` dégèle les 30 dernières couches. Or VGG19 n'a que 22 couches : pour lui, **toute** la base serait
dégelée (20 millions de paramètres), alors que MobileNetV2 n'en dégèle que 1,5 million. Le protocole ne serait donc
pas vraiment le même pour les trois modèles. À discuter avant de lancer l'étape 4.

## 3. Notions théoriques

**Transfer learning.** Les premières couches d'un CNN apprennent des motifs très généraux (bords, coins, textures),
utiles pour n'importe quelle image. On réutilise donc un modèle déjà entraîné sur ImageNet au lieu de tout réapprendre
avec nos 4 447 radios. Image simple : un médecin qui apprend à lire des radios sait déjà voir, il n'a pas à réapprendre
ce qu'est un contour.

**Backbone et tête.** Le *backbone* (ou base) est le CNN pré-entraîné : il transforme une image en caractéristiques.
La *tête* est la petite partie ajoutée à la fin, qui transforme ces caractéristiques en réponse. On a retiré la tête
d'origine (1 000 classes ImageNet) pour mettre la nôtre (1 sortie).

**Couches gelées.** Leurs poids ne changent pas pendant l'entraînement. C'est rapide (1 281 poids à ajuster au lieu
de 2,2 millions) et cela évite de détruire ce que le modèle sait déjà.

**Global Average Pooling.** Le backbone sort 1 280 petites cartes de 7×7. On remplace chaque carte par sa moyenne :
on obtient un vecteur de 1 280 nombres, un par caractéristique. C'est beaucoup moins de paramètres qu'un `Flatten`.

**Dropout.** Pendant l'entraînement, 30 % des valeurs sont mises à zéro au hasard. Le modèle ne peut pas s'appuyer sur
une seule caractéristique : cela limite le surapprentissage. Il est désactivé pour la validation et la prédiction.

**Sigmoïde.** Elle transforme n'importe quel nombre en une valeur entre 0 et 1, qu'on lit comme la probabilité de pneumonie.

**Binary cross-entropy.** La loss standard pour 2 classes. Elle est petite quand la probabilité prédite est proche de
la vraie étiquette, et très grande quand le modèle est sûr de lui mais se trompe.

**Adam et learning rate.** L'optimiseur corrige les poids pour faire baisser la loss. Le *learning rate* est la taille
du pas : trop grand, on dépasse le minimum ; trop petit, on avance trop lentement. Adam adapte le pas de chaque poids.

**Epoch.** Un passage complet sur les 4 447 images d'entraînement, soit 139 lots de 32.

**Callbacks.** Des fonctions appelées automatiquement à la fin de chaque epoch : arrêter, sauvegarder, réduire le
learning rate, enregistrer l'historique.

**Surapprentissage (overfitting).** Le modèle apprend le train par cœur : sa loss de train baisse encore alors que
celle de validation remonte. Ce n'est pas le cas ici.

## 4. Fonctions et classes

### Créées

| Nom | Fichier | Rôle | Paramètres |
|---|---|---|---|
| `BACKBONES` | `models.py` | Dictionnaire : nom → classe Keras du modèle pré-entraîné | 3 clés |
| `LAST_CONV` | `models.py` | Dictionnaire : nom → dernière couche convolutionnelle (pour Grad-CAM) | 3 clés |
| `build_model(name, dropout)` | `models.py` | Construit le modèle complet, base gelée ; renvoie `(modèle, base)` | `name` : le modèle ; `dropout` : 0,3 par défaut |
| `compile_model(model, lr)` | `models.py` | Choisit l'optimiseur, la loss et les métriques | `lr` : learning rate |
| `fine_tune(base, n_unfreeze)` | `models.py` | Dégèle les dernières couches (étape 4) | `n_unfreeze` : 30 par défaut |
| `callbacks(name, phase)` | `train.py` | Renvoie les 4 callbacks | nom du modèle, `"phase1"` ou `"phase2"` |
| `tracer_courbes(name)` | `train.py` | Trace loss et accuracy depuis les CSV | nom du modèle |
| `entrainer(name, epochs1, epochs2, n_unfreeze)` | `train.py` | Enchaîne phase 1, phase 2, sauvegarde et courbes | epochs de chaque phase |

### Utilisées

| Nom | Rôle | Paramètres importants |
|---|---|---|
| `keras.applications.MobileNetV2 / ResNet50 / VGG19` | Charge un CNN pré-entraîné | `include_top=False`, `weights="imagenet"`, `input_tensor` |
| `layers.GlobalAveragePooling2D` | Résume chaque carte par sa moyenne | aucun |
| `layers.Dropout` | Met à zéro une part des valeurs pendant l'entraînement | le taux (0,3) |
| `layers.Dense(1, activation="sigmoid")` | Neurone de sortie : probabilité de pneumonie | nombre de neurones, activation |
| `model.compile` | Configure l'apprentissage | `optimizer`, `loss`, `metrics` |
| `model.fit` | Entraîne | `validation_data`, `epochs`, `class_weight`, `callbacks` |
| `model.save` / `keras.models.load_model` | Sauvegarde / recharge un modèle `.keras` | le chemin |
| `EarlyStopping` | Arrête si la validation ne progresse plus | `monitor`, `patience=5`, `restore_best_weights` |
| `ModelCheckpoint` | Sauvegarde le meilleur modèle | `monitor`, `save_best_only` |
| `ReduceLROnPlateau` | Réduit le learning rate si la validation stagne | `factor=0.2`, `patience=2`, `min_lr` |
| `CSVLogger` | Écrit les métriques de chaque epoch dans un CSV | le chemin |
| `keras.utils.set_random_seed` | Fixe la seed de Python, NumPy et TensorFlow | la seed |
| `argparse.ArgumentParser` | Lit les options de la ligne de commande | `--model`, `--epochs1`, `--epochs2` |

## 5. Le code important, expliqué ligne par ligne

### 5.1 Construire le modèle (`models.py`)

```python
def build_model(name, dropout=0.3):
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    base = BACKBONES[name](include_top=False, weights="imagenet",
                           input_tensor=inputs, input_shape=IMG_SIZE + (3,))
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=name), base
```

- `keras.Input(shape=IMG_SIZE + (3,))` : l'entrée du modèle, une image 224×224 à 3 canaux. `(224, 224) + (3,)` donne `(224, 224, 3)`.
- `BACKBONES[name](...)` : on va chercher la classe du modèle dans le dictionnaire et on la construit.
  - `include_top=False` : on retire la dernière couche d'ImageNet (1 000 classes).
  - `weights="imagenet"` : on charge les poids déjà appris (téléchargés une seule fois).
  - `input_tensor=inputs` : les couches du backbone sont branchées directement sur notre entrée. Elles font alors
    partie de notre modèle, et on pourra retrouver `out_relu` par son nom pour Grad-CAM.
- `base.trainable = False` : on gèle toutes les couches du backbone.
- `GlobalAveragePooling2D()(base.output)` : le bloc 7×7×1280 devient un vecteur de 1 280 valeurs.
- `Dropout(dropout)(x)` : 30 % de ces valeurs sont éteintes au hasard pendant l'entraînement.
- `Dense(1, activation="sigmoid")(x)` : un seul neurone, qui sort la probabilité de PNEUMONIA.
  Il a 1 280 poids + 1 biais = **1 281 paramètres**, les seuls entraînés en phase 1.
- `return keras.Model(inputs, outputs, name=name), base` : on renvoie le modèle complet, et la base à part pour
  pouvoir la dégeler à l'étape 4.

### 5.2 Compiler (`models.py`)

```python
def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")])
```

- `optimizer=Adam(learning_rate=lr)` : la méthode de correction des poids et la taille du pas.
- `loss="binary_crossentropy"` : ce que le modèle cherche à minimiser.
- `metrics=[...]` : ce qu'on affiche pour suivre l'entraînement. Elles ne servent pas à apprendre, seulement à observer.
  On suit le recall et l'AUC en plus de l'accuracy, car les classes sont déséquilibrées.

### 5.3 Les callbacks (`train.py`)

```python
def callbacks(name, phase):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(MODELS_DIR / f"{name}_{phase}.keras"),
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
        keras.callbacks.CSVLogger(str(MODELS_DIR / f"{name}_{phase}_history.csv")),
    ]
```

- `EarlyStopping` : surveille la loss de validation. Si elle ne s'améliore pas pendant 5 epochs, l'entraînement
  s'arrête. `restore_best_weights=True` remet les poids de la meilleure epoch.
- `ModelCheckpoint` : enregistre le modèle à chaque fois que la loss de validation bat son record.
- `ReduceLROnPlateau` : si la loss de validation stagne 2 epochs, le learning rate est multiplié par 0,2 (divisé par 5).
  Des pas plus petits permettent d'affiner.
- `CSVLogger` : écrit une ligne par epoch dans un fichier CSV, qui sert ensuite à tracer les courbes.
- Les quatre surveillent la **validation**, jamais le test.

### 5.4 Entraîner (`train.py`)

```python
keras.utils.set_random_seed(SEED)

def entrainer(name, epochs1=15, epochs2=10, n_unfreeze=30):
    MODELS_DIR.mkdir(exist_ok=True)
    train, val, _ = get_datasets(name)
    cw = class_weights()
    model, base = build_model(name)

    compile_model(model, lr=1e-3)
    model.fit(train, validation_data=val, epochs=epochs1,
              class_weight=cw, callbacks=callbacks(name, "phase1"))

    if epochs2 > 0:
        fine_tune(base, n_unfreeze)
        compile_model(model, lr=1e-5)
        model.fit(train, validation_data=val, epochs=epochs2,
                  class_weight=cw, callbacks=callbacks(name, "phase2"))

    model.save(MODELS_DIR / f"{name}.keras")
    tracer_courbes(name)
```

- `set_random_seed(SEED)` : fixe tous les tirages aléatoires (initialisation de la tête, dropout, mélange).
- `train, val, _ = get_datasets(name)` : on récupère le pipeline de l'étape 2. Le `_` signifie qu'on ignore
  volontairement le troisième élément, le **test**.
- `cw = class_weights()` : les poids `{0: 1.94, 1: 0.67}`.
- `compile_model(model, lr=1e-3)` : learning rate de 0,001, assez élevé car la tête part de zéro.
- `model.fit(...)` : l'entraînement. `validation_data=val` évalue à la fin de chaque epoch,
  `class_weight=cw` pondère la loss, `callbacks=...` branche les 4 surveillants.
- `if epochs2 > 0` : la phase 2 (fine-tuning) n'est faite que si on le demande. Avec `--epochs2 0`, on s'arrête
  après la phase 1 : c'est ce qu'on a fait à cette étape.
- `model.save(...)` : enregistre le modèle final dans `models/mobilenetv2.keras`.
- `tracer_courbes(name)` : produit la figure des courbes.

### 5.5 Tracer les courbes (`train.py`)

```python
h = pd.concat(historiques, ignore_index=True)
epochs = range(1, len(h) + 1)
ax.plot(epochs, h[metrique], color="#2a78d6", linewidth=2, marker="o", label="train")
ax.plot(epochs, h[f"val_{metrique}"], color="#eb6834", linewidth=2, marker="o", label="validation")
```

- `pd.concat` : met bout à bout l'historique de la phase 1 et, s'il existe, celui de la phase 2.
- `range(1, len(h) + 1)` : numérote les epochs à partir de 1.
- Les deux `plot` tracent la même métrique pour le train (bleu) et la validation (orange).
  Dans le CSV, la colonne de validation porte le même nom précédé de `val_`.

### 5.6 Les options de la ligne de commande (`train.py`)

```python
p = argparse.ArgumentParser(description="Entraîne un modèle de détection de pneumonie.")
p.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "resnet50", "vgg19"])
p.add_argument("--epochs1", type=int, default=15)
p.add_argument("--epochs2", type=int, default=10)
a = p.parse_args()
entrainer(a.model, a.epochs1, a.epochs2)
```

- `add_argument` déclare une option, sa valeur par défaut et, pour `--model`, les seules valeurs acceptées.
- `parse_args()` lit ce qui a été tapé. `python src/train.py --model resnet50` donne `a.model == "resnet50"`.
- Le même script sert donc aux trois modèles, sans modifier le code.

## 6. Trois questions de jury

**Q1. Pourquoi le transfer learning plutôt qu'un CNN entraîné de zéro, alors qu'ImageNet ne contient pas de radios ?**
Avec 4 447 images, un CNN partant de zéro surapprendrait : il a des millions de poids à régler. Les premières couches
d'un modèle pré-entraîné détectent des motifs généraux (bords, textures, formes) qui restent utiles sur des radios,
même si ImageNet contient des photos. En gelant la base, on n'entraîne que 1 281 paramètres, et on obtient 95,5 %
d'accuracy en validation en quelques epochs. La limite est réelle : les couches profondes sont spécialisées sur des
objets du quotidien. C'est justement pour cela que l'étape 4 les réadapte par fine-tuning.

**Q2. Sur vos courbes, la validation est meilleure que l'entraînement. Est-ce une erreur ?**
Non, et il y a trois explications. D'abord, les métriques du train sont mesurées avec le dropout actif et sur des
images augmentées, donc dans des conditions plus difficiles que la validation. Ensuite, la loss du train est pondérée
par les poids de classes, pas celle de validation : les deux ne sont pas directement comparables. Enfin, notre
validation contient des patients dont d'autres radios sont dans le train, ce qui la rend un peu trop facile.
Ce qui compte, c'est que la loss de validation ne remonte pas : il n'y a pas de surapprentissage.

**Q3. À quoi servent vos callbacks, et pourquoi surveiller la loss de validation plutôt que l'accuracy ?**
`EarlyStopping` arrête l'entraînement quand la validation ne progresse plus et restaure les meilleurs poids,
`ModelCheckpoint` sauvegarde le meilleur modèle, `ReduceLROnPlateau` réduit le learning rate quand on stagne, et
`CSVLogger` garde l'historique pour les courbes. On surveille la loss car elle est plus fine que l'accuracy :
l'accuracy ne change que quand une prédiction franchit le seuil de 0,5, alors que la loss tient compte de la confiance
du modèle. Chez nous, l'accuracy de validation est restée presque identique de l'epoch 8 à l'epoch 15 (entre 95,4 %
et 95,7 %), alors que la loss a continué de varier et a désigné l'epoch 11 comme la meilleure.
