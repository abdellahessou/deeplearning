# Étape 5 – Évaluation complète

## 1. Résumé

On a évalué en détail le modèle retenu à l'étape 4, **MobileNetV2**, sur les 624 radios du **test set**. Le nouveau
script `src/evaluate.py` produit le rapport de classification, la matrice de confusion, la courbe ROC, la courbe
précision-rappel, un tableau qui montre l'effet du seuil de décision, et deux grilles d'images : les pneumonies ratées
et les fausses alertes dont le modèle était le plus sûr. En médecine, rater une pneumonie est plus grave qu'une fausse
alerte : le **recall** est la métrique clé. Résultat : le modèle rate très peu de pneumonies (5 sur 390), mais déclenche
beaucoup de fausses alertes (93 radios normales sur 234).

## 2. Résultat obtenu

Commande : `python src/evaluate.py --model mobilenetv2` (environ 1 minute sur le PC). Le script a aussi été lancé sur
ResNet50 et VGG19 pour vérifier qu'il fonctionne avec les trois modèles.

### Rapport de classification (seuil 0,5) – `reports/rapport_mobilenetv2.txt`

| Classe | Precision | Recall | F1 | Nombre d'images |
|---|---|---|---|---|
| NORMAL | 0,966 | 0,603 | 0,742 | 234 |
| PNEUMONIA | 0,805 | **0,987** | 0,887 | 390 |
| Moyenne simple (macro) | 0,886 | 0,795 | 0,815 | 624 |
| Moyenne pondérée | 0,866 | 0,843 | 0,833 | 624 |

Accuracy : **84,3 %**. Mêmes chiffres qu'à l'étape 4, ce qui confirme que les deux scripts sont cohérents.

**Comment le lire.** Chaque ligne regarde le problème du point de vue d'une classe.

- **Ligne PNEUMONIA** : recall 0,987, le modèle trouve 98,7 % des pneumonies ; precision 0,805, quand il annonce une
  pneumonie, il a raison 4 fois sur 5.
- **Ligne NORMAL** : recall 0,603, il ne reconnaît que 60 % des radios normales (c'est la **spécificité**) ;
  precision 0,966, quand il annonce « normal », il a raison dans 96,6 % des cas. C'est rassurant : un résultat
  « normal » est fiable.
- **Moyenne simple (macro)** : la moyenne des deux classes, sans tenir compte de leur taille. Elle est plus basse que la
  moyenne pondérée, car elle donne autant de poids à la classe NORMAL, la moins bien traitée.

### Matrice de confusion – `reports/figures/cm_mobilenetv2.png`

|  | Prédit NORMAL | Prédit PNEUMONIA |
|---|---|---|
| **Réellement NORMAL** (234) | 141 vrais négatifs (60,3 %) | **93 faux positifs** (39,7 %) |
| **Réellement PNEUMONIA** (390) | **5 faux négatifs** (1,3 %) | 385 vrais positifs (98,7 %) |

**Comment la lire.** Les lignes sont la vérité, les colonnes la réponse du modèle. La diagonale (en haut à gauche, en
bas à droite) contient les bonnes réponses ; le reste, les erreurs. Les pourcentages sont calculés par ligne : 39,7 %
des radios normales sont classées pneumonie. Les erreurs sont **très déséquilibrées** : 93 fausses alertes pour
5 pneumonies ratées. Le modèle penche nettement vers « pneumonie ».

### Courbe ROC – `reports/figures/roc_mobilenetv2.png`

**AUC = 0,962.**

**Comment la lire.** Chaque point de la courbe correspond à un seuil. En abscisse, la part des radios normales
déclarées malades (fausses alertes) ; en ordonnée, la part des pneumonies détectées (recall). Le coin en haut à gauche
serait parfait : toutes les pneumonies trouvées, aucune fausse alerte. La diagonale grise est un modèle qui répond au
hasard. Notre courbe monte très vite et reste collée en haut : le modèle **classe bien** les radios. Le point orange
(seuil 0,5) est en haut à droite de la courbe : recall de 98,7 % mais 40 % de fausses alertes. En se déplaçant vers la
gauche sur la courbe (seuil plus haut), on garderait un recall autour de 95 % avec seulement 15 % de fausses alertes.

### Courbe précision-rappel – `reports/figures/pr_mobilenetv2.png`

**AP (aire sous la courbe) = 0,975**, contre 0,625 pour un modèle au hasard.

**Comment la lire.** En abscisse le recall, en ordonnée la precision, encore un point par seuil. La ligne grise est le
niveau du hasard : un modèle qui dirait toujours « pneumonie » aurait une precision de 0,625, la part de pneumonies dans
le test. Notre precision reste proche de 1 jusqu'à un recall d'environ 0,7, puis baisse. Le point orange est dans le
« coude » de la fin : pour aller chercher les dernières pneumonies, le modèle paie en fausses alertes. Cette courbe est
utile quand les classes sont déséquilibrées, car elle ne regarde que la classe PNEUMONIA.

### Effet du seuil de décision – `reports/seuils_mobilenetv2.csv`

| Seuil | Recall | Precision | Spécificité | Accuracy | Pneumonies ratées (FN) | Fausses alertes (FP) |
|---|---|---|---|---|---|---|
| 0,3 | 0,992 | 0,782 | 0,538 | 82,2 % | 3 | 108 |
| 0,4 | 0,990 | 0,794 | 0,573 | 83,3 % | 4 | 100 |
| **0,5** | 0,987 | 0,805 | 0,603 | 84,3 % | 5 | 93 |
| 0,6 | 0,982 | 0,822 | 0,645 | 85,6 % | 7 | 83 |
| 0,7 | 0,979 | 0,849 | 0,709 | 87,8 % | 8 | 68 |

**Comment le lire.** Plus on monte le seuil, plus le modèle doit être sûr de lui pour annoncer une pneumonie :
les fausses alertes baissent (108 → 68), mais les pneumonies ratées augmentent (3 → 8). Il n'existe pas de seuil
« gratuit » : c'est un **compromis**. Passer de 0,5 à 0,7 évite 25 fausses alertes au prix de 3 pneumonies ratées
en plus.

Pour aller plus loin (calculé à part, hors du tableau du guide) : avec un seuil de 0,9, on aurait 16 pneumonies
ratées et 45 fausses alertes (accuracy 90,2 %). L'accuracy monte, mais on rate trois fois plus de malades : en
médecine, ce n'est pas forcément un progrès.

### Les erreurs les plus confiantes

**Faux négatifs – `reports/figures/erreurs_fn_mobilenetv2.png`.** Il n'y en a que **5**, la grille en montre donc 5
et non 8. Ce sont 5 pneumonies bactériennes, avec des probabilités de 0,15 à 0,41 : le modèle n'en était pas
« très sûr », sauf pour les deux premières (0,15 et 0,16). Juger si ces pneumonies sont réellement difficiles à voir
demanderait l'avis d'un radiologue ; Grad-CAM (étape 6) montrera au moins où le modèle a regardé.

**Faux positifs – `reports/figures/erreurs_fp_mobilenetv2.png`** (ajout par rapport au guide, car c'est l'erreur la
plus fréquente). Le modèle est presque certain de se tromper : probabilités de 0,996 à 1,000. Ces radios ont l'air
normales. On remarque que **7 des 8 portent un nom en `NORMAL2-`** et plusieurs ont un marquage « KV58 / KV59 » dans
un coin. Une vérification sur tout le test le confirme :

| Radios normales du test | Nombre | Fausses alertes | Taux |
|---|---|---|---|
| Fichiers `IM-…` | 69 | 16 | 23 % |
| Fichiers `NORMAL2-IM-…` | 165 | 77 | **47 %** |

Les fausses alertes viennent surtout du groupe `NORMAL2`. Le train contient aussi des fichiers de ce groupe
(633 sur 1 147 radios normales), donc ce n'est pas un groupe totalement nouveau. Mais ces radios ont peut-être été
prises dans d'autres conditions (appareil, réglages, période) que celles vues à l'entraînement. Ce n'est qu'une
hypothèse : le dataset ne documente pas l'origine de ces noms. Elle rejoint l'écart validation / test constaté à
l'étape 4.

### Les trois modèles (même script)

| Modèle | Accuracy | Recall | Spécificité | FN | FP | AUC | AP |
|---|---|---|---|---|---|---|---|
| **MobileNetV2** | 84,3 % | 0,987 | 0,603 | 5 | 93 | 0,962 | 0,975 |
| ResNet50 | 83,2 % | 0,974 | 0,594 | 10 | 95 | 0,950 | 0,965 |
| VGG19 | 84,1 % | 0,990 | 0,594 | 4 | 95 | 0,968 | 0,980 |

Les trois font le même type d'erreurs, en même quantité. Le choix de MobileNetV2 fait à l'étape 4 reste valable.

### Fichiers produits

- `src/evaluate.py` : le script d'évaluation (nouveau).
- `reports/rapport_<modèle>.txt` : le rapport de classification.
- `reports/seuils_<modèle>.csv` : le tableau des seuils.
- `reports/figures/cm_<modèle>.png`, `roc_<modèle>.png`, `pr_<modèle>.png` : matrice de confusion et courbes.
- `reports/figures/erreurs_fn_<modèle>.png`, `erreurs_fp_<modèle>.png` : les grilles d'erreurs.

Ces fichiers existent pour les trois modèles ; le rapport utilisera ceux de MobileNetV2.

### Écarts par rapport au guide

- **Courbe précision-rappel et grilles d'erreurs** : demandées dans le prompt du guide mais absentes de son code de
  référence ; elles ont été ajoutées.
- **Grille des faux positifs en plus de celle des faux négatifs**, car ce sont eux qui font baisser l'accuracy.
- **Le tableau des seuils est enregistré dans un CSV** et contient aussi la spécificité et l'accuracy.
- **`argparse`** (`--model`) au lieu de `sys.argv`, comme dans `train.py`.
- **`compile=False`** au chargement du modèle, pour la raison expliquée à l'étape 4.
- **Matrice de confusion avec les pourcentages** de chaque classe réelle, en plus des nombres.

### Vérifications faites

- Les chiffres (accuracy, precision, recall, F1, AUC) sont identiques à ceux de `compare.py` à l'étape 4.
- Le script vérifie que chaque fichier image correspond bien à son étiquette (le nom du dossier `NORMAL` ou
  `PNEUMONIA` est comparé à l'étiquette lue par le pipeline), pour que les grilles montrent les bonnes radios.
- La grille des faux négatifs fonctionne quand il y a moins de 8 erreurs (5 ici).

### Limites à connaître

- **Le seuil ne doit pas être choisi en regardant le test.** Le tableau des seuils sert à *comprendre* le compromis.
  Si on choisissait 0,7 parce qu'il donne la meilleure accuracy sur le test, ce chiffre deviendrait optimiste. Il
  faudrait choisir le seuil sur la validation, mais la nôtre est trop facile (patients communs avec le train, étape 1).
  On garde donc 0,5, et l'application de l'étape 7 laissera l'utilisateur régler le seuil lui-même.
- **624 images, dont 234 normales** : chaque pourcentage a une marge d'incertitude de quelques points.
- **L'hypothèse sur les fichiers `NORMAL2`** n'est pas vérifiable avec les informations du dataset.

## 3. Notions théoriques

**Les quatre cas possibles.** Pour chaque radio, la vérité et la réponse du modèle donnent l'un de ces cas :

| | Le modèle dit NORMAL | Le modèle dit PNEUMONIA |
|---|---|---|
| **Vraiment NORMAL** | Vrai négatif (TN) | Faux positif (FP) : fausse alerte |
| **Vraiment PNEUMONIA** | Faux négatif (FN) : malade raté | Vrai positif (TP) |

« Positif » veut dire « le modèle annonce la maladie », pas « bonne nouvelle ».

**Recall (sensibilité) = TP / (TP + FN).** Parmi les vrais malades, la part détectée. Ici 385 / 390 = 98,7 %.
C'est la métrique clé : un malade raté rentre chez lui sans traitement.

**Precision = TP / (TP + FP).** Parmi les alertes, la part justifiée. Ici 385 / 478 = 80,5 %.

**Spécificité = TN / (TN + FP).** Parmi les personnes saines, la part reconnue comme saine. Ici 141 / 234 = 60,3 %.
C'est le recall de la classe NORMAL.

**F1-score.** La moyenne harmonique de la precision et du recall : 2 × P × R / (P + R). Elle est basse dès que l'une
des deux est basse, contrairement à une moyenne simple.

**Accuracy = (TP + TN) / total.** La part de bonnes réponses. Elle peut tromper quand les classes sont déséquilibrées :
un modèle qui dirait toujours « pneumonie » aurait déjà 62,5 % d'accuracy sur notre test.

**Seuil de décision.** Le modèle sort une probabilité entre 0 et 1. Le seuil est la valeur à partir de laquelle on
annonce une pneumonie. 0,5 est l'habitude, pas une obligation. Baisser le seuil = détecter plus de malades et
déclencher plus de fausses alertes.

**Courbe ROC et AUC.** La courbe ROC trace le recall en fonction du taux de fausses alertes, pour **tous** les seuils.
L'AUC est l'aire sous cette courbe. Elle a une lecture simple : si on tire au hasard une radio malade et une radio
saine, l'AUC est la probabilité que le modèle donne une probabilité plus haute à la malade. 0,962 signifie qu'il les
range dans le bon ordre 96 fois sur 100. Elle ne dépend pas du seuil.

**Courbe précision-rappel et AP.** Même idée, mais avec la precision au lieu du taux de fausses alertes. L'AP
(*average precision*) est l'aire sous cette courbe. Elle est plus sévère que l'AUC quand la classe positive est rare,
parce qu'elle ne compte pas les vrais négatifs.

**Analyse d'erreurs.** Regarder les images mal classées, surtout celles où le modèle était très sûr de lui. C'est
souvent là qu'on découvre un biais des données (ici, les fichiers `NORMAL2`) plutôt qu'un défaut du modèle.

## 4. Fonctions et classes

### Créées

| Nom | Rôle | Paramètres |
|---|---|---|
| `predire(name)` | Charge le modèle, prédit sur le test, renvoie vraies étiquettes, probabilités et chemins des images | nom du modèle |
| `matrice_confusion(y_true, y_pred, name)` | Dessine la matrice de confusion, renvoie le tableau TN/FP/FN/TP | étiquettes vraies et prédites, nom |
| `courbe_roc(y_true, y_prob, name, seuil)` | Dessine la courbe ROC avec l'AUC et le point du seuil choisi | `seuil` : 0,5 par défaut |
| `courbe_precision_rappel(y_true, y_prob, name, seuil)` | Dessine la courbe précision-rappel, renvoie l'AP | `seuil` : 0,5 par défaut |
| `tableau_seuils(y_true, y_prob, seuils)` | Calcule recall, precision, spécificité, accuracy, FN et FP pour chaque seuil | `seuils` : 0,3 à 0,7 |
| `grille_erreurs(y_true, y_prob, chemins, name, type_erreur, n, seuil)` | Affiche les n erreurs les plus confiantes | `type_erreur` : `"fn"` ou `"fp"` ; `n` : 8 |

### Utilisées

| Nom | Rôle | Paramètres importants |
|---|---|---|
| `classification_report` | Precision, recall, F1 par classe, plus les moyennes | `target_names`, `digits=3` |
| `confusion_matrix` | Compte TN, FP, FN, TP | `y_true`, `y_pred` |
| `roc_curve`, `auc` | Points de la courbe ROC et aire sous la courbe | `y_true`, `y_prob` |
| `precision_recall_curve` | Points de la courbe précision-rappel | `y_true`, `y_prob` |
| `average_precision_score` | Aire sous la courbe précision-rappel (AP) | `y_true`, `y_prob` |
| `sns.heatmap` | Tableau coloré (matrice de confusion) | `annot`, `fmt`, `cmap` |
| `Dataset.file_paths` | Liste des fichiers lus par `image_dataset_from_directory` | aucun |
| `np.where`, `np.argsort` | Trouver les indices des erreurs, puis les trier | condition ; valeurs à trier |
| `PIL.Image.open(...).convert("L")` | Ouvre une image d'origine en niveaux de gris | le chemin |

## 5. Le code important, expliqué ligne par ligne

### 5.1 Prédire sur le test (`predire`)

```python
_, _, test = get_datasets(name)
model = keras.models.load_model(MODELS_DIR / f"{name}.keras", compile=False)
y_true = np.concatenate([y.numpy() for _, y in test]).ravel().astype(int)
y_prob = model.predict(test, verbose=0).ravel()
chemins = np.array(charger_split("test", shuffle=False).file_paths)
return y_true, y_prob, chemins
```

- `_, _, test = get_datasets(name)` : on ne garde que le **test**, avec le prétraitement propre au modèle.
- `load_model(..., compile=False)` : on recharge l'architecture et les poids, sans les réglages d'entraînement.
- `y_true` : on parcourt les lots, on garde les étiquettes (`y`), on les colle bout à bout et on obtient 624 valeurs 0 ou 1.
- `y_prob` : 624 probabilités de pneumonie. `.ravel()` transforme le tableau 624×1 en simple liste de 624 valeurs.
- `chemins` : le chemin de chaque image. Le test n'est jamais mélangé (`shuffle=False`), donc la 10e probabilité
  correspond au 10e fichier. On en a besoin pour afficher les radios mal classées.

### 5.2 La matrice de confusion

```python
cm = confusion_matrix(y_true, y_pred)
pct = cm / cm.sum(axis=1, keepdims=True) * 100
textes = [[f"{cm[i, j]}\n({pct[i, j]:.1f} %)" for j in range(2)] for i in range(2)]
sns.heatmap(cm, annot=textes, fmt="", cmap="Blues", cbar=False, ax=ax,
            xticklabels=CLASSES, yticklabels=CLASSES)
```

- `confusion_matrix` renvoie `[[TN, FP], [FN, TP]]` : ligne = vraie classe, colonne = classe prédite.
- `cm.sum(axis=1, keepdims=True)` : le total de chaque ligne (234 normales, 390 pneumonies). En divisant, on obtient
  le pourcentage de chaque case dans sa ligne.
- `textes` : pour chaque case, le nombre puis le pourcentage entre parenthèses (`\n` = retour à la ligne).
- `sns.heatmap(...)` : colore chaque case selon son nombre (bleu foncé = beaucoup). `annot=textes` écrit nos textes,
  `fmt=""` dit à seaborn de les afficher tels quels.

### 5.3 La courbe ROC

```python
fpr, tpr, _ = roc_curve(y_true, y_prob)
y_pred = y_prob >= seuil
point = (y_pred[y_true == 0].mean(), y_pred[y_true == 1].mean())
ax.plot(fpr, tpr, color=BLEU, linewidth=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Hasard (AUC = 0,5)")
ax.plot(*point, "o", color=ORANGE, markersize=9, label=...)
```

- `roc_curve` essaie tous les seuils possibles et renvoie, pour chacun, le taux de faux positifs (`fpr`) et le taux de
  vrais positifs (`tpr`, le recall).
- `y_pred[y_true == 0].mean()` : parmi les radios normales, la part déclarée malade = taux de faux positifs au seuil 0,5.
  `y_pred[y_true == 1].mean()` : parmi les pneumonies, la part détectée = recall. Ce sont les coordonnées du point orange.
- Premier `plot` : la courbe, et l'AUC calculée par `auc(fpr, tpr)` dans la légende.
- Deuxième `plot` : la diagonale du hasard, en pointillés.
- `*point` « déplie » le couple : `ax.plot(*point, ...)` revient à `ax.plot(x, y, ...)`.

### 5.4 Le tableau des seuils

```python
for s in seuils:
    y_pred = (y_prob >= s).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    lignes.append({"seuil": s, "recall": tp / (tp + fn), "precision": tp / (tp + fp),
                   "spécificité": tn / (tn + fp), "accuracy": (tp + tn) / len(y_true),
                   "FN": fn, "FP": fp})
```

- Pour chaque seuil, on reconvertit les probabilités en classes 0 / 1.
- `(y_pred == 1) & (y_true == 1)` : vrai pour chaque radio prédite malade **et** vraiment malade. `.sum()` compte les
  vrais : c'est TP. Les trois lignes suivantes font de même pour FN, FP et TN.
- On applique ensuite les formules de la section 3 et on range le tout dans un dictionnaire, une ligne du tableau.

### 5.5 La grille des erreurs les plus confiantes

```python
if type_erreur == "fn":
    idx = np.where((y_true == 1) & (y_prob < seuil))[0]
    idx = idx[np.argsort(y_prob[idx])][:n]
else:
    idx = np.where((y_true == 0) & (y_prob >= seuil))[0]
    idx = idx[np.argsort(-y_prob[idx])][:n]
for ax, i in zip(axes.flat, idx):
    img = Image.open(chemins[i]).convert("L")
    ax.imshow(img, cmap="gray")
```

- `np.where(...)[0]` : les numéros des radios qui sont des faux négatifs (malades, mais probabilité sous le seuil).
- `np.argsort(y_prob[idx])` : l'ordre qui trie ces radios par probabilité croissante. La plus basse est celle où le
  modèle était le plus sûr que la radio était normale : l'erreur la plus « confiante ».
- `[:n]` : on garde les 8 premières (ou moins s'il y en a moins).
- Pour les faux positifs, on trie par `-y_prob` : le signe moins inverse l'ordre, les plus hautes probabilités d'abord.
- `Image.open(chemins[i]).convert("L")` : on rouvre l'image **d'origine** (non redimensionnée, non prétraitée) en
  niveaux de gris, pour la montrer telle qu'un médecin la verrait.

### 5.6 Une vérification de sécurité

```python
assert all(CLASSES[t] in c.replace("\\", "/").split("/")[-2] for t, c in zip(y_true, chemins))
```

- Pour chaque image, `split("/")[-2]` donne le nom de son dossier (`NORMAL` ou `PNEUMONIA`).
- `CLASSES[t]` donne le nom de la classe d'après l'étiquette du pipeline.
- `assert all(...)` arrête le script si un seul couple ne correspond pas. Cela garantit que les probabilités, les
  étiquettes et les fichiers sont bien alignés.

## 6. Trois questions de jury

**Q1. Votre accuracy est de 84 %. Pourquoi dites-vous que le recall est plus important ?**
Parce que les deux erreurs n'ont pas le même coût. Un faux négatif, c'est un enfant atteint de pneumonie qu'on renvoie
chez lui sans traitement. Un faux positif, c'est un examen de contrôle en plus par un médecin. Notre modèle rate
5 pneumonies sur 390 (recall de 98,7 %) : c'est le chiffre qui compte pour un outil de dépistage. L'accuracy de 84 % est
tirée vers le bas par les 93 fausses alertes. Elle mélange les deux types d'erreurs comme s'ils se valaient, ce qui
n'est pas le cas ici.

**Q2. Pourquoi ne pas simplement monter le seuil à 0,7, puisque l'accuracy passe à 87,8 % ?**
Pour deux raisons. D'abord, cela fait rater 3 pneumonies de plus (8 au lieu de 5) : on gagne en accuracy mais on perd
sur la métrique la plus importante. Ensuite, ce chiffre a été mesuré sur le test : choisir le seuil en regardant le
test, c'est ajuster le modèle sur les données qui doivent servir à le juger, et le résultat devient trop optimiste. Le
seuil devrait être choisi sur la validation. Nous avons gardé 0,5 et l'application permettra au médecin de régler
lui-même le compromis.

**Q3. L'AUC est de 0,96, mais 40 % des radios normales sont mal classées. Comment est-ce possible ?**
L'AUC mesure si le modèle range bien les radios, de la plus saine à la plus malade, quel que soit le seuil. Il les
range bien : dans 96 % des paires, la radio malade reçoit une probabilité plus haute que la saine. Mais les
probabilités sont décalées vers le haut : la moitié des radios normales du test reçoivent plus de 0,23, et un quart
plus de 0,78. Le seuil de 0,5 coupe donc au mauvais endroit pour ce test. L'analyse d'erreurs montre que ces radios
viennent surtout d'un sous-groupe de fichiers (`NORMAL2`) : le problème vient plus des données que du modèle.
