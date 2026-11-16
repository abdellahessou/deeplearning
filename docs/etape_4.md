# Étape 4 – Fine-tuning et comparaison des 3 modèles

## 1. Résumé

On a entraîné les trois modèles (MobileNetV2, ResNet50, VGG19) avec exactement le même protocole en deux phases :
d'abord la base gelée, puis un **fine-tuning** qui réentraîne doucement le dernier bloc de la base. L'entraînement
a été fait sur Google Colab (GPU T4), car le PC n'a pas de GPU utilisable. Un nouveau script, `src/compare.py`,
évalue ensuite les trois modèles sur le **test set** et produit un tableau et un graphique de comparaison.
Résultat : les trois modèles sont à égalité sur le test (accuracy ≈ 84 %, recall 97 à 99 %), donc on retient
**MobileNetV2**, dix fois plus léger et le plus rapide.

## 2. Résultat obtenu

### Entraînement (Colab, GPU T4, TensorFlow 2.21.0)

Commande, identique pour les trois modèles : `python src/train.py --model <modèle> --epochs1 15 --epochs2 10`.

Le tableau donne, pour chaque phase, la **meilleure epoch** (plus petite loss de validation). Ce sont des chiffres de
**validation** (785 images) : ils servent à piloter l'entraînement, pas à juger le modèle.

| Modèle | Phase | Epochs faites | Meilleure epoch | Loss val | Accuracy val | Precision val | Recall val | AUC val |
|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | 1 – base gelée | 15 | 13 | 0,104 | 96,2 % | 98,4 % | 96,4 % | 0,991 |
| MobileNetV2 | 2 – fine-tuning | 6 | 1 | **0,093** | 96,4 % | 98,8 % | 96,4 % | 0,993 |
| ResNet50 | 1 – base gelée | 8 | 3 | 0,105 | 95,4 % | 98,8 % | 95,0 % | 0,993 |
| ResNet50 | 2 – fine-tuning | 6 | 1 | **0,084** | 96,8 % | 99,5 % | 96,2 % | 0,997 |
| VGG19 | 1 – base gelée | 11 | 6 | 0,131 | 95,2 % | 99,1 % | 94,3 % | 0,990 |
| VGG19 | 2 – fine-tuning | 7 | 2 | **0,084** | 96,6 % | 99,5 % | 95,9 % | 0,997 |

Ce qu'on y lit :

- **Le fine-tuning améliore la loss de validation des trois modèles** : de 0,104 à 0,093 (MobileNetV2), de 0,105 à
  0,084 (ResNet50), de 0,131 à 0,084 (VGG19). Le gain est le plus net pour VGG19, qui était le moins bon en phase 1.
- **L'arrêt anticipé s'est déclenché partout en phase 2** (6 ou 7 epochs faites sur 10 possibles), et aussi en phase 1
  pour ResNet50 et VGG19. Le modèle sauvegardé est celui de la meilleure epoch, pas celui de la dernière.
- **La meilleure epoch du fine-tuning est la 1re ou la 2e.** Ensuite la validation ne progresse plus.

### Comment lire les courbes (`reports/figures/courbes_<modèle>.png`)

Le trait vertical en pointillés marque le début du fine-tuning.

- **Avant le trait**, les deux courbes se stabilisent : la tête a appris ce qu'elle pouvait avec une base gelée.
- **Après le trait**, la loss d'entraînement se remet à descendre franchement (ResNet50 : de 0,110 à 0,041 ;
  VGG19 : de 0,136 à 0,035 ; MobileNetV2 : de 0,119 à 0,079), parce que des millions de poids sont maintenant ajustables.
- **Mais la loss de validation ne suit pas** : elle fait des dents de scie (ResNet50 : 0,084 puis 0,164, 0,185, 0,109…).
  L'écart qui se creuse entre train et validation est un **début de surapprentissage**. C'est exactement le cas où
  `EarlyStopping` est utile : il a arrêté l'entraînement et remis les poids de la meilleure epoch.
- `ReduceLROnPlateau` a réduit le learning rate du fine-tuning de 0,00001 à 0,000002 puis 0,0000004.

### Comparaison sur le test set (`python src/compare.py`)

624 images jamais vues pendant l'entraînement, seuil de décision 0,5.

| Modèle | Accuracy | Precision | Recall | F1 | AUC | Paramètres | Fichier `.keras` | Temps par image (CPU du PC) |
|---|---|---|---|---|---|---|---|---|
| **MobileNetV2** | 84,3 % | 80,5 % | 98,7 % | 0,887 | 0,962 | 2,26 M | 22 Mo | ≈ 10 ms |
| ResNet50 | 83,2 % | 80,0 % | 97,4 % | 0,879 | 0,950 | 23,59 M | 210 Mo | ≈ 29 ms |
| VGG19 | 84,1 % | 80,2 % | 99,0 % | 0,886 | 0,968 | 20,02 M | 156 Mo | ≈ 92 ms |

Le même tableau est dans `reports/comparaison.csv`, et le graphique dans `reports/figures/comparaison_modeles.png`
(F1 en bleu, recall en orange).

Détail des erreurs, déduit du tableau (390 radios PNEUMONIA et 234 radios NORMAL dans le test) :

| Modèle | Pneumonies détectées | Pneumonies ratées (faux négatifs) | Normales bien classées | Normales classées pneumonie (faux positifs) |
|---|---|---|---|---|
| MobileNetV2 | 385 | 5 | 141 | 93 |
| ResNet50 | 380 | 10 | 139 | 95 |
| VGG19 | 386 | 4 | 139 | 95 |

### Ce que disent ces chiffres

1. **Les trois modèles sont à égalité.** Entre le meilleur et le moins bon, il y a 7 images sur 624 (526 bonnes
   réponses contre 519). Avec 624 images, l'accuracy est connue à environ ± 3 points près : un écart de 1 point ne
   permet pas de dire qu'un modèle est meilleur qu'un autre.
2. **Très peu de pneumonies ratées** : 4 à 10 sur 390. C'est l'erreur la plus grave en médecine, et elle est rare.
3. **Beaucoup de fausses alertes** : environ 94 radios normales sur 234 (40 %) sont classées PNEUMONIA. C'est ce qui
   fait tomber l'accuracy à 84 %.
4. **L'AUC reste élevée (0,95 à 0,97)** : les modèles donnent bien des probabilités plus fortes aux pneumonies qu'aux
   radios normales. Le problème vient surtout du **seuil de 0,5**, trop bas pour ce test. L'étape 5 étudiera ce seuil.

### Pourquoi 96 % en validation et 84 % sur le test ?

L'écart est grand et il faut savoir l'expliquer :

- **La validation est trop facile.** Elle a été tirée du même dossier que le train, et 362 de ses patients atteints de
  pneumonie ont aussi des radios dans le train (limite notée à l'étape 1). Le modèle revoit donc des patients connus.
- **Le test est un lot à part**, fourni tel quel par les auteurs du dataset, avec une autre proportion de classes :
  62,5 % de pneumonies contre 74,2 % dans le train. Le modèle, habitué à voir surtout des pneumonies, en prédit trop.
- C'est pour cela que la règle du projet impose de mesurer les résultats finaux **sur le test** : c'est le seul chiffre
  honnête. Annoncer 96 % serait trompeur.

### Le fine-tuning a-t-il servi ?

- **Sur la validation, oui** : la loss baisse pour les trois modèles (tableau du haut).
- **Sur le test, on ne voit pas de gain pour MobileNetV2**, le seul modèle pour lequel on a une mesure « avant » :

| MobileNetV2 | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Phase 1 seule (modèle de l'étape 3, entraîné sur le PC) | 84,3 % | 80,4 % | 99,0 % | 0,887 | 0,962 |
| Après fine-tuning (entraîné sur Colab) | 84,3 % | 80,5 % | 98,7 % | 0,887 | 0,962 |

  Les deux lignes sont identiques à une image près. Attention : ce sont deux entraînements différents (PC et Colab),
  donc la comparaison est indicative. Pour ResNet50 et VGG19, on n'a pas gardé le modèle de fin de phase 1 : on ne peut
  pas mesurer leur « avant / après » sur le test.
- **Explication probable** : le fine-tuning s'est arrêté très tôt (meilleure epoch 1 ou 2) avec un learning rate très
  faible. Les poids ont donc peu bougé. Et la principale difficulté du test, le décalage avec le train, ne se corrige
  pas en réentraînant sur les mêmes images.

### Modèle retenu : MobileNetV2

À performance égale, on prend le plus simple : 2,26 millions de paramètres contre 20 à 24 millions, un fichier de 22 Mo
contre 156 à 210 Mo, et le temps de réponse le plus court (environ 3 fois plus rapide que ResNet50 et 9 fois plus que
VGG19). C'est aussi le plus adapté à l'application Streamlit de
l'étape 7. Les étapes 5 et 6 pourront utiliser les trois modèles, mais MobileNetV2 sera le modèle principal.

### Fichiers produits

- `src/compare.py` : le script de comparaison (nouveau).
- `src/models.py` : ajout de `N_UNFREEZE` ; `src/train.py` : la phase 2 utilise `N_UNFREEZE[name]`.
- `notebooks/colab_train.ipynb` : le notebook Colab, exécuté sur GPU T4.
- `models/mobilenetv2.keras`, `resnet50.keras`, `vgg19.keras` : les trois modèles finaux (non versionnés, trop lourds).
- `models/<modèle>_phase1_history.csv` et `_phase2_history.csv` : les six historiques.
- `reports/figures/courbes_<modèle>.png` : les trois figures de courbes, avec les deux phases.
- `reports/comparaison.csv` et `reports/figures/comparaison_modeles.png` : la comparaison sur le test.

### Vérifications faites

- **Fine-tuning** (mini-entraînement des trois modèles sur le PC) : le learning rate passe bien de 0,001 à 0,00001 ;
  aucune couche BatchNorm n'est entraînable et leurs statistiques sont identiques avant et après la phase 2 ; les
  couches modifiées par la phase 2 sont exactement celles qu'on a dégelées.
- **Modèles rapportés de Colab** : chaque modèle, rechargé sur le PC et réévalué sur la validation, redonne exactement
  les chiffres de la meilleure epoch de sa phase 2 (loss 0,0928 / 0,0838 / 0,0837). Les fichiers sont donc les bons, et
  ils contiennent bien les poids de la meilleure epoch.
- **`compare.py`** : exécuté sur les trois modèles ; les nombres d'erreurs déduits du tableau redonnent bien l'accuracy
  affichée (par exemple MobileNetV2 : (385 + 141) / 624 = 84,3 %).

### Écarts par rapport au guide

- **Nombre de couches dégelées par modèle.** Le guide dégèle 30 couches partout. Or VGG19 n'en a que 22 : toute sa base
  (20 millions de paramètres) aurait été réentraînée, contre 1,5 million pour MobileNetV2. On dégèle donc à peu près
  le dernier bloc de chaque modèle :

| Modèle | Couches dégelées | Ce que cela représente | Paramètres entraînables en phase 2 |
|---|---|---|---|
| MobileNetV2 | 30 | blocs 14 à 16 et dernière convolution | 1 512 001 (67 %) |
| ResNet50 | 30 | l'étage `conv5` | 14 430 721 (61 %) |
| VGG19 | 5 | le bloc 5 (4 convolutions) | 9 439 745 (47 %) |

- **`compare.py` mesure aussi le temps par image** et le nombre de paramètres, pour pouvoir départager des modèles
  à performance égale.
- **Les modèles sont rechargés avec `compile=False`.** Un fichier `.keras` garde les réglages d'entraînement. Sur Colab,
  Keras active une option d'accélération prévue pour le GPU (`jit_compile=True`). Rechargée telle quelle sur le PC, elle
  rendait les prédictions 4 fois plus lentes (MobileNetV2 : 35 ms par image au lieu de 9), sans changer les résultats.
  Avec `compile=False`, on ne recharge que l'architecture et les poids, ce qui suffit pour prédire. Il faudra faire de
  même aux étapes 5, 6 et 7.

### Limites à connaître

- **Temps par image mesuré sur le processeur du PC**, par lots de 32, sans GPU. Il dépend de la machine et de ce
  qu'elle fait au même moment : c'est un ordre de grandeur, utile pour comparer les modèles entre eux.
- **Un seul entraînement par modèle.** Avec une autre seed, les chiffres bougeraient un peu. Les écarts entre modèles
  sont plus petits que cette variation.
- **Choisir un modèle en regardant le test** peut biaiser le résultat en notre faveur. Ici le risque est limité : les
  scores sont à égalité et le choix se fait sur la taille et la vitesse, pas sur le score.
- **Les courbes de MobileNetV2 ont été refaites.** `courbes_mobilenetv2.png` et `mobilenetv2_phase1_history.csv`
  viennent maintenant de l'entraînement Colab. Les chiffres de `docs/etape_3.md` décrivent l'entraînement fait sur le
  PC : ils sont très proches (loss de validation 0,105 contre 0,104) mais pas identiques.

## 3. Notions théoriques

**Fine-tuning.** Après avoir entraîné la tête, on « dégèle » les dernières couches de la base et on continue
l'entraînement. Ces couches, spécialisées sur les objets d'ImageNet (chiens, voitures…), s'adaptent alors aux radios.
Image simple : le médecin qui savait déjà voir affine maintenant son œil sur des radios.

**Pourquoi seulement les dernières couches ?** Les premières couches détectent des motifs généraux (bords, textures)
qui servent pour toutes les images : on les garde. Les dernières détectent des formes propres à ImageNet : ce sont
elles qu'il faut adapter. Dégeler tout le réseau avec 4 447 images ferait surapprendre.

**Pourquoi deux phases ?** Au début, la tête a des poids aléatoires, donc ses erreurs sont énormes. Si la base était
dégelée dès le départ, ces grosses corrections abîmeraient les poids pré-entraînés. On entraîne donc d'abord la tête
seule, puis on dégèle.

**Pourquoi un learning rate très faible (0,00001) ?** Les poids pré-entraînés sont déjà bons : on veut les retoucher,
pas les réécrire. On fait donc des pas cent fois plus petits qu'en phase 1.

**Pourquoi recompiler ?** Changer `trainable` ne prend effet qu'après un nouveau `compile` : c'est à ce moment que
Keras fixe la liste des poids à entraîner et le nouvel optimiseur.

**BatchNorm gelées.** Une couche BatchNorm garde en mémoire la moyenne et l'écart-type de ce qui la traverse, calculés
sur ImageNet. Si on la laissait s'entraîner, elle recalculerait ces statistiques sur des lots de 32 radios : elles
seraient instables et casseraient ce que le réseau a appris. On les laisse donc gelées. MobileNetV2 en a 52,
ResNet50 53, VGG19 aucune.

**Surapprentissage.** Le modèle retient le train par cœur : sa loss de train baisse, celle de validation stagne ou
remonte. On l'a observé pendant le fine-tuning, et l'arrêt anticipé l'a limité.

**Precision, recall, F1.**
- *Recall* : parmi les vraies pneumonies, combien le modèle en trouve. Un recall de 98,7 % = 5 pneumonies ratées sur 390.
- *Precision* : parmi les radios annoncées PNEUMONIA, combien le sont vraiment. 80,5 % = 1 alerte sur 5 est fausse.
- *F1* : une moyenne (harmonique) des deux, qui n'est élevée que si les deux le sont.

**AUC.** Elle mesure si le modèle range bien les radios, de la plus saine à la plus malade, **sans dépendre du seuil**.
1 = classement parfait, 0,5 = hasard. Une AUC élevée avec une accuracy moyenne signifie : bon classement, seuil mal réglé.

**Seuil de décision.** Le modèle sort une probabilité. On annonce PNEUMONIA si elle dépasse 0,5. Ce 0,5 est un choix,
pas une loi : le monter réduit les fausses alertes mais fait rater plus de pneumonies.

**Écart validation / test (généralisation).** Un modèle peut être excellent sur des données qui ressemblent à
celles de l'entraînement et moins bon sur des données un peu différentes. Le test mesure la vraie capacité à généraliser.

**Temps d'inférence.** Le temps qu'il faut au modèle pour répondre sur une image. Il compte pour une application réelle.

## 4. Fonctions et classes

### Créées ou modifiées

| Nom | Fichier | Rôle | Paramètres |
|---|---|---|---|
| `N_UNFREEZE` | `models.py` | Dictionnaire : nom du modèle → nombre de couches à dégeler | 3 clés (30, 30, 5) |
| `fine_tune(base, n_unfreeze)` | `models.py` | Dégèle les `n` dernières couches, garde les BatchNorm gelées | `base` : le backbone ; `n_unfreeze` : nombre de couches |
| `entrainer(name, epochs1, epochs2)` | `train.py` | Phase 1, puis phase 2 avec `N_UNFREEZE[name]`, sauvegarde, courbes | nom du modèle, epochs max de chaque phase |
| `tracer_courbes(name)` | `train.py` | Trace les deux phases bout à bout, avec le trait du fine-tuning | nom du modèle |
| `evaluer(name, seuil)` | `compare.py` | Évalue un modèle sur le test, renvoie un dictionnaire de métriques | `seuil` : 0,5 par défaut |
| `tracer_comparaison(df)` | `compare.py` | Diagramme en barres F1 / recall par modèle | `df` : le tableau des résultats |

### Utilisées

| Nom | Rôle | Paramètres importants |
|---|---|---|
| `layer.trainable` | Indique si les poids d'une couche peuvent changer | `True` / `False` |
| `layers.BatchNormalization` | Type de couche qu'on garde gelée | utilisé avec `isinstance` |
| `keras.models.load_model` | Recharge un modèle `.keras` | le chemin, `compile=False` |
| `model.predict` | Calcule les probabilités | `batch_size`, `verbose=0` |
| `model.count_params` | Nombre total de paramètres | aucun |
| `time.perf_counter` | Chronomètre précis | aucun |
| `np.concatenate` | Met bout à bout les lots en un seul tableau | la liste de tableaux |
| `accuracy_score`, `precision_score`, `recall_score`, `f1_score` | Métriques de scikit-learn | `y_true`, `y_pred` (classes 0 / 1) |
| `roc_auc_score` | AUC | `y_true`, `y_prob` (probabilités, pas classes) |
| `pd.DataFrame`, `DataFrame.to_csv` | Tableau de résultats et export CSV | `index=False` |
| `ax.bar`, `ax.bar_label` | Barres et valeurs écrites au-dessus | position, hauteur, largeur, `fmt` |

## 5. Le code important, expliqué ligne par ligne

### 5.1 Dégeler les dernières couches (`models.py`)

```python
N_UNFREEZE = {"mobilenetv2": 30, "resnet50": 30, "vgg19": 5}

def fine_tune(base, n_unfreeze=30):
    base.trainable = True
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
```

- `N_UNFREEZE` : le nombre de couches à dégeler pour chaque modèle (voir « Écarts par rapport au guide »).
- `base.trainable = True` : on rend d'abord toute la base entraînable.
- `base.layers[:-n_unfreeze]` : toutes les couches **sauf** les `n` dernières. Le `-` compte depuis la fin.
  On regèle ces couches-là : il ne reste donc que les `n` dernières d'entraînables.
- La deuxième boucle parcourt toutes les couches : `isinstance(layer, layers.BatchNormalization)` vaut `True` si la
  couche est une BatchNorm, et on la gèle, même si elle fait partie des `n` dernières.

### 5.2 La phase 2 dans l'entraînement (`train.py`)

```python
if epochs2 > 0:
    fine_tune(base, N_UNFREEZE[name])
    compile_model(model, lr=1e-5)
    model.fit(train, validation_data=val, epochs=epochs2,
              class_weight=cw, callbacks=callbacks(name, "phase2"))

model.save(MODELS_DIR / f"{name}.keras")
```

- `fine_tune(base, N_UNFREEZE[name])` : dégèle le dernier bloc du modèle choisi.
- `compile_model(model, lr=1e-5)` : recompilation obligatoire, avec un learning rate 100 fois plus petit qu'en phase 1.
- `model.fit(...)` : mêmes données, mêmes poids de classes, mêmes callbacks qu'en phase 1. Seuls changent les couches
  entraînables, le learning rate et le nom des fichiers (`phase2`).
- `epochs=epochs2` : 10 epochs **au maximum** ; `EarlyStopping` a arrêté après 6 ou 7.
- `model.save(...)` : le modèle final. Comme `EarlyStopping` a restauré les meilleurs poids, c'est le modèle de la
  meilleure epoch de la phase 2.

### 5.3 Évaluer un modèle sur le test (`compare.py`)

```python
def evaluer(name, seuil=0.5):
    _, _, test = get_datasets(name)
    model = keras.models.load_model(MODELS_DIR / f"{name}.keras", compile=False)

    x = np.concatenate([images.numpy() for images, _ in test])
    y_true = np.concatenate([labels.numpy() for _, labels in test]).ravel().astype(int)
```

- `_, _, test = get_datasets(name)` : cette fois on ignore le train et la validation, et on ne garde que le **test**.
  Chaque modèle reçoit son propre prétraitement (`name`).
- `load_model(..., compile=False)` : recharge le modèle entraîné sur Colab, sans ses réglages d'entraînement
  (optimiseur, loss). Pour prédire, l'architecture et les poids suffisent, et c'est plus rapide sur le PC.
- `for images, _ in test` : on parcourt les lots de 32 ; `np.concatenate` les colle en un seul tableau de 624 images.
- `y_true` : les vraies étiquettes. `.ravel()` aplatit le tableau, `.astype(int)` donne des 0 et des 1.

```python
    model.predict(x[:BATCH_SIZE], verbose=0)
    t0 = time.perf_counter()
    y_prob = model.predict(x, batch_size=BATCH_SIZE, verbose=0).ravel()
    ms_image = (time.perf_counter() - t0) / len(y_true) * 1000
```

- Premier `predict` : un tour de chauffe sur 32 images. Le premier appel est toujours plus lent (TensorFlow prépare
  ses calculs) : on ne le chronomètre pas.
- `t0 = time.perf_counter()` : on lance le chronomètre.
- `y_prob` : les 624 probabilités de pneumonie.
- `ms_image` : temps écoulé, divisé par 624 images, multiplié par 1 000 pour passer des secondes aux millisecondes.

```python
    y_pred = (y_prob >= seuil).astype(int)
    return {"modèle": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob),
            "paramètres (M)": model.count_params() / 1e6,
            "ms / image": ms_image}
```

- `y_pred = (y_prob >= seuil).astype(int)` : 1 (PNEUMONIA) si la probabilité atteint le seuil, sinon 0 (NORMAL).
- Les quatre premières métriques comparent les classes prédites (`y_pred`) aux vraies.
- `roc_auc_score(y_true, y_prob)` : l'AUC reçoit les **probabilités**, pas les classes, car elle ne dépend pas du seuil.
- `model.count_params() / 1e6` : le nombre de paramètres, en millions.

### 5.4 Le programme principal (`compare.py`)

```python
noms = [n for n in PREPROCESS if (MODELS_DIR / f"{n}.keras").exists()]
df = pd.DataFrame([evaluer(n) for n in noms]).round(3)
df.to_csv(FIG_DIR.parent / "comparaison.csv", index=False)
tracer_comparaison(df)
```

- `noms` : on ne garde que les modèles dont le fichier `.keras` existe. Le script fonctionne donc avec 1, 2 ou 3 modèles.
- `[evaluer(n) for n in noms]` : une liste de dictionnaires ; `pd.DataFrame` en fait un tableau, une ligne par modèle.
- `.round(3)` : trois chiffres après la virgule.
- `to_csv(..., index=False)` : écrit le tableau, sans la colonne de numéros de ligne.

### 5.5 Le graphique (`compare.py`)

```python
for i, (metrique, couleur) in enumerate([("f1", "#2a78d6"), ("recall", "#eb6834")]):
    barres = ax.bar(x + (i - 0.5) * (largeur + 0.02), df[metrique], largeur, label=metrique, color=couleur)
    ax.bar_label(barres, fmt="%.3f", padding=3)
ax.set_ylim(0, 1.2)
```

- La boucle trace deux séries de barres : le F1 en bleu, le recall en orange.
- `x + (i - 0.5) * (largeur + 0.02)` : décale la première série vers la gauche et la deuxième vers la droite, pour
  qu'elles soient côte à côte.
- `bar_label` : écrit la valeur au-dessus de chaque barre.
- `set_ylim(0, 1.2)` : l'axe part de **0**. Couper l'axe à 0,8 ferait paraître les écarts bien plus grands qu'ils ne sont.

## 6. Trois questions de jury

**Q1. Pourquoi 96 % d'accuracy en validation mais seulement 84 % sur le test ? Votre modèle a-t-il surappris ?**
Ce n'est pas un surapprentissage classique, c'est surtout un problème de données. Notre validation vient du même
dossier que le train et partage des patients avec lui : elle est trop facile. Le test est un lot séparé, avec moins de
pneumonies (62,5 % contre 74,2 %). Le modèle en prédit donc trop : il rate très peu de malades (recall 98,7 %) mais
classe à tort 40 % des radios normales en pneumonie. L'AUC de 0,96 montre que le classement des radios reste bon :
c'est le seuil de 0,5 qui n'est pas adapté. Le chiffre honnête à annoncer est celui du test, 84 %.

**Q2. Pourquoi geler les couches BatchNorm pendant le fine-tuning, et pourquoi un learning rate aussi faible ?**
Une BatchNorm mémorise la moyenne et l'écart-type calculés sur ImageNet. Si on la laisse s'entraîner, elle les
recalcule sur des lots de 32 radios : ces statistiques sont instables et perturbent tout le réseau. On a vérifié, sur
un mini-entraînement, qu'aucune des 52 BatchNorm de MobileNetV2 ne change pendant la phase 2. Quant au learning rate
de 0,00001, il sert à retoucher des poids déjà bons sans les détruire : avec 0,001, on risquerait d'effacer en quelques
lots ce que le modèle a appris sur 1,2 million d'images.

**Q3. Quel modèle avez-vous retenu, et pourquoi pas le plus gros ?**
MobileNetV2. Sur le test, les trois modèles sont à égalité : 84,3 %, 83,2 % et 84,1 % d'accuracy, soit 7 images
d'écart sur 624, moins que la marge d'incertitude (± 3 points). Quand la performance est la même, on prend le plus
simple : MobileNetV2 a 2,3 millions de paramètres contre 20 à 24 millions, son fichier pèse 22 Mo contre 156 à 210 Mo,
et il répond le plus vite. Un gros modèle n'est pas forcément meilleur : avec seulement 4 447 images d'entraînement,
ses millions de paramètres supplémentaires n'ont rien apporté ici et augmentent le risque de surapprentissage.
