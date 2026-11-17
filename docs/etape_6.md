# Étape 6 – Explicabilité avec Grad-CAM

## 1. Résumé

Un CNN donne une probabilité mais n'explique pas pourquoi : c'est une « boîte noire ». **Grad-CAM** produit une carte de
chaleur qui montre les zones de la radio qui ont le plus pesé dans la décision. On a écrit `src/gradcam.py` avec deux
fonctions, `make_gradcam_heatmap` (calcule la carte) et `superposer` (la colore et la pose sur la radio avec OpenCV),
qui fonctionnent pour les trois modèles. Le script produit une figure avec quatre cas du test (vrai positif, vrai
négatif, faux positif, faux négatif) et une figure sur les fausses alertes de l'étape 5. Résultat : MobileNetV2 regarde
surtout le thorax, mais pas toujours les poumons eux-mêmes, et les autres modèles s'appuient parfois sur des zones
sans intérêt médical.

## 2. Résultat obtenu

Commande : `python src/gradcam.py --model mobilenetv2` (environ 25 secondes sur le PC, environ 0,4 seconde par carte).
Le script a aussi été lancé pour ResNet50 et VGG19.

### Les quatre cas – `reports/figures/gradcam_<modèle>.png`

Pour chaque type de cas, on prend l'image du test où le modèle était **le plus sûr de lui**. En haut la radio, en bas la
radio avec la carte : **rouge = zone très importante** pour la décision, bleu = zone peu importante. La carte explique
toujours **la classe prédite** : pour une radio prédite NORMAL, elle montre ce qui a fait pencher vers « normal ».

Ce qu'on voit pour **MobileNetV2** :

| Cas | Probabilité | Où sont les zones rouges |
|---|---|---|
| Vrai positif | 1,000 | Sur les deux poumons et le centre du thorax : c'est le comportement attendu. |
| Vrai négatif | 0,000 | Surtout en bas, au centre (cœur et coupole du diaphragme). |
| Faux positif | 1,000 | Une tache en bas du médiastin et une bande le long du poumon gauche de l'image. |
| Faux négatif | 0,154 | En haut du thorax et à la base du cou, plus une tache sous le diaphragme (abdomen) : pas sur les poumons. |

Comparaison avec les deux autres modèles (même script) :

- **VGG19** : sur le faux positif, une **tache rouge recouvre la lettre « R »** en haut à gauche de la radio. Ce marqueur
  n'a rien à voir avec la maladie. C'est un exemple du biais « Clever Hans » décrit dans le guide. Sur le vrai positif,
  un coin en bas à droite est aussi coloré.
- **ResNet50** : sur le vrai positif, les zones les plus rouges sont **sur les bords de l'image**, en dehors des poumons.
  Sur le vrai négatif, elles sont sur le cou et les clavicules.
- **Sur une même radio, les trois modèles ne regardent pas les mêmes zones**, alors qu'ils donnent la même réponse.

### Les fausses alertes – `reports/figures/gradcam_fp_<modèle>.png`

À l'étape 5, on avait vu que les fausses alertes les plus confiantes venaient surtout des fichiers `NORMAL2-…` et
portaient parfois un marquage « KV59 » dans un coin. Avec MobileNetV2, pour ces 4 radios :

- les zones rouges sont **dans le thorax** : centre et bas du médiastin, bas des deux poumons pour la n° 4 ;
- **le marquage « KV59 » n'est pas mis en avant** (le coin en bas à droite reste bleu) ;
- pour la n° 2, la tache la plus rouge est en haut à gauche, dans le poumon, près de la lettre « R » sans la couvrir ;
  pour la n° 4, la zone autour du « R » est légèrement colorée.

On ne peut donc pas dire que MobileNetV2 « lit » le marquage. Les fausses alertes viennent plutôt de l'aspect général
de ces radios (contraste, densité), ce que Grad-CAM ne permet pas de préciser.

### Un indicateur sur les 624 radios du test (MobileNetV2)

Huit images ne suffisent pas à conclure. On a donc calculé la carte de **toutes** les radios du test et mesuré quelle
part de la « chaleur » tombe dans deux zones :

| Zone | Part de la surface de l'image | Part de la carte de chaleur (moyenne) |
|---|---|---|
| Carré central (là où se trouve le thorax) | 36 % | **46 %** |
| Bande de 16 pixels le long des bords (là où se trouvent les lettres et marquages) | 27 % | **17 %** |

Le modèle se concentre plus que le hasard sur le centre et moins que le hasard sur les bords. **Aucune** des 624 radios
n'a plus de la moitié de sa carte sur les bords. C'est un indicateur grossier (le carré central n'est pas un masque des
poumons), mais il rassure : MobileNetV2 ne s'appuie pas systématiquement sur les bords ou le texte.

### Une correction importante par rapport au guide

Le code du guide calcule le gradient de la **probabilité** (après la sigmoïde). Quand le modèle est sûr à 100 %, la
sigmoïde est « saturée » : la probabilité vaut exactement 1,0 et son gradient est **nul**. La carte est alors vide.

Vérification sur la radio la plus confiante de chaque modèle :

| Modèle | Probabilité | Carte avec le code du guide | Carte avec notre version |
|---|---|---|---|
| MobileNetV2 | 1,0000000 | **vide** (gradient = 0, 0 case sur 49) | normale (49 cases) |
| VGG19 | 1,0000000 | **vide** (gradient = 0, 0 case sur 196) | normale |
| ResNet50 | 0,9999991 | correcte, mais gradient minuscule (2,6 × 10⁻⁹) | normale |

VGG19 a 8 radios du test dans ce cas. Notre version calcule le gradient du **logit**, le score juste avant la sigmoïde,
qui ne sature jamais. La carte reste la même quand la probabilité n'est pas saturée : seule l'échelle des gradients
change, et la normalisation finale l'efface.

### Fichiers produits

- `src/gradcam.py` : le code (nouveau).
- `reports/figures/gradcam_<modèle>.png` : les 4 cas, pour les 3 modèles.
- `reports/figures/gradcam_fp_<modèle>.png` : les 4 faux positifs les plus confiants, pour les 3 modèles.

### Vérifications faites

- **Même lecture d'image que le pipeline** : la probabilité calculée par `gradcam.py` est identique à celle de
  l'étape 5 (écart maximal 1 × 10⁻⁶ sur 20 radios tirées au hasard, pour chacun des 3 modèles).
- **Taille de la carte** : 7×7 pour MobileNetV2 et ResNet50, 14×14 pour VGG19, comme prévu à l'étape 3.
- **Les 4 cas sont les bons** : ils correspondent aux probabilités et aux faux négatifs trouvés à l'étape 5
  (par exemple le faux négatif de MobileNetV2, `person154_bacteria_728.jpeg`, P = 0,154).

### Écarts par rapport au guide

- **Gradient du logit au lieu de la probabilité** (voir plus haut).
- **La carte explique la classe prédite** : le guide calcule toujours la carte de « pneumonie » et propose `1 - p`
  en remarque. Ici, un paramètre `classe` choisit le sens ; pour la classe NORMAL, on prend `-logit`, ce qui revient
  au même que `1 - p` sans le problème de saturation.
- **Lecture de l'image comme le pipeline** (`tf.image.resize`, interpolation bilinéaire), pour que la carte explique
  exactement la prédiction de l'étape 5.
- **Figure supplémentaire sur les faux positifs**, pour suivre la piste ouverte à l'étape 5.
- `keras.Model(model.input, …)` au lieu de `model.inputs` : la version du guide déclenche un avertissement de Keras 3.

### Limites à connaître

- **Carte très grossière** : 7×7 cases pour une radio de 224×224, chaque case couvre 32×32 pixels. L'agrandissement
  lisse la carte, mais il n'ajoute aucune précision. On ne peut pas localiser une petite opacité.
- **Corrélation, pas preuve** : une zone rouge a influencé le score, mais cela ne veut pas dire que le modèle y a
  « vu » une pneumonie.
- **On ne montre que les cas extrêmes** (les plus confiants) : ils ne sont pas forcément représentatifs.
- **Aucune vérification par un médecin** : dire si une zone est « médicalement pertinente » demanderait un radiologue
  ou des masques des poumons, que le dataset ne fournit pas.

## 3. Notions théoriques

**Explicabilité (XAI).** Rendre compréhensible la décision d'un modèle. En médecine, c'est indispensable : un médecin
ne peut pas faire confiance à un outil qui répond « pneumonie » sans montrer sur quoi il se base.

**Cartes de caractéristiques.** La dernière couche convolutionnelle de MobileNetV2 sort 1 280 petites images de 7×7.
Chacune répond à un motif (une texture, une forme) et garde l'information « où » : la case en haut à gauche correspond
au haut à gauche de la radio. C'est la dernière couche qui sait encore *où* sont les choses. Après, le Global Average
Pooling fait la moyenne et perd la position.

**Gradient.** Il mesure de combien la sortie bouge quand on modifie un peu une valeur. Un gradient fort sur une case
veut dire : si cette zone changeait, le score changerait beaucoup. Elle compte donc pour la décision.

**Logit.** Le nombre que sort le dernier neurone avant la sigmoïde (de −∞ à +∞). La sigmoïde le transforme en
probabilité (de 0 à 1). Quand le logit est très grand, la probabilité est collée à 1 et ne bouge presque plus :
c'est la **saturation**. Le logit, lui, continue de varier.

**L'algorithme Grad-CAM, étape par étape.**

1. On fait passer la radio dans le modèle et on garde deux choses : les 1 280 cartes 7×7 de la dernière convolution,
   et le score final (le logit).
2. On calcule le gradient du score par rapport à chaque case de chaque carte (`tf.GradientTape`) : 1 280 × 7 × 7 nombres.
3. Pour chaque carte, on fait la moyenne de ses 49 gradients. On obtient **un poids par carte** : son importance pour
   la décision. Une carte avec un poids positif pousse vers la classe expliquée.
4. On additionne les 1 280 cartes, chacune multipliée par son poids. On obtient une seule carte 7×7.
5. On met à zéro les valeurs négatives (**ReLU**) : on ne garde que ce qui pousse *vers* la classe, pas ce qui pousse
   contre.
6. On divise par le maximum pour avoir des valeurs entre 0 et 1.
7. On agrandit la carte à 224×224, on la colore (bleu → rouge) et on la pose en transparence sur la radio.

**Biais « Clever Hans ».** Clever Hans était un cheval qui semblait savoir compter : en fait, il lisait les réactions
de son maître. Un modèle peut faire pareil : obtenir un bon score en s'appuyant sur un indice parasite (une lettre,
un marquage, un tube) qui est lié à la classe dans les données. Grad-CAM aide à le repérer : on l'a vu une fois avec
VGG19 sur la lettre « R ».

## 4. Fonctions et classes

### Créées

| Nom | Rôle | Paramètres |
|---|---|---|
| `make_gradcam_heatmap(model, img_array, backbone, classe)` | Calcule la carte Grad-CAM, valeurs entre 0 et 1 | `img_array` : image prétraitée (1, 224, 224, 3) ; `backbone` : nom du modèle ; `classe` : 1 = PNEUMONIA, 0 = NORMAL |
| `superposer(img_rgb, heatmap, alpha)` | Agrandit, colore et superpose la carte sur la radio | `img_rgb` : radio uint8 (224, 224, 3) ; `alpha` : transparence, 0,4 |
| `charger_image(chemin)` | Lit une radio comme le pipeline (RGB, 224×224, bilinéaire) | le chemin |
| `expliquer(model, backbone, chemin)` | Probabilité + radio + radio avec la carte de la classe prédite | modèle, nom, chemin |
| `figure(model, backbone, cas, fichier, titre)` | Figure : une colonne par cas, radio en haut, Grad-CAM en bas | `cas` : liste (nom, chemin, vraie classe) |

### Utilisées

| Nom | Rôle | Paramètres importants |
|---|---|---|
| `model.get_layer` | Récupère une couche par son nom | `LAST_CONV[backbone]` |
| `keras.Model(entrée, [sorties])` | Construit un modèle qui renvoie des sorties intermédiaires | entrée et liste de sorties |
| `tf.GradientTape` | Enregistre les calculs pour obtenir des gradients | `tape.gradient(score, conv_out)` |
| `tf.matmul` | Produit matriciel (calcul du logit) | vecteur de caractéristiques, poids de la couche Dense |
| `tf.reduce_mean`, `tf.reduce_sum`, `tf.maximum` | Moyenne, somme, ReLU | `axis` |
| `tf.io.decode_image`, `tf.image.resize` | Lire et redimensionner une image | `channels=3`, taille |
| `cv2.resize` | Agrandit la carte 7×7 en 224×224 | taille (largeur, hauteur) |
| `cv2.applyColorMap` | Transforme des valeurs 0–255 en couleurs | `cv2.COLORMAP_JET` |
| `cv2.cvtColor` | Convertit BGR (OpenCV) en RGB (matplotlib) | `cv2.COLOR_BGR2RGB` |

## 5. Le code important, expliqué ligne par ligne

### 5.1 Un modèle qui renvoie les sorties intermédiaires

```python
conv_layer = model.get_layer(LAST_CONV[backbone])
dense = model.layers[-1]
grad_model = keras.Model(model.input, [conv_layer.output, dense.input])
```

- `LAST_CONV[backbone]` : le nom de la dernière convolution (`out_relu` pour MobileNetV2), défini à l'étape 3.
  C'est grâce à `input_tensor` (étape 3) que cette couche fait partie de notre modèle et qu'on peut la retrouver par
  son nom.
- `model.layers[-1]` : la dernière couche, le `Dense(1, sigmoid)`.
- `grad_model` : un nouveau modèle, qui partage les mêmes couches et les mêmes poids, mais renvoie deux choses :
  les cartes de la convolution et le vecteur qui entre dans la couche Dense.

### 5.2 Le gradient du logit

```python
with tf.GradientTape() as tape:
    conv_out, features = grad_model(img_array, training=False)
    logit = tf.matmul(features, dense.kernel) + dense.bias
    score = logit[:, 0] if classe == 1 else -logit[:, 0]
grads = tape.gradient(score, conv_out)
```

- `with tf.GradientTape() as tape:` : TensorFlow « enregistre » tous les calculs faits dans ce bloc, pour pouvoir
  ensuite remonter les gradients.
- `grad_model(img_array, training=False)` : on fait passer la radio. `training=False` désactive le dropout.
- `tf.matmul(features, dense.kernel) + dense.bias` : on refait à la main le calcul de la couche Dense, **sans** la
  sigmoïde : c'est le logit. `kernel` contient les 1 280 poids, `bias` le biais.
- `score` : le logit pour expliquer PNEUMONIA, son opposé pour expliquer NORMAL.
- `tape.gradient(score, conv_out)` : le gradient du score par rapport à chaque valeur des cartes. Même forme que
  `conv_out` : (1, 7, 7, 1280).

### 5.3 De 1 280 cartes à une seule

```python
poids = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_sum(conv_out[0] * poids, axis=-1)
heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
return heatmap.numpy()
```

- `reduce_mean(grads, axis=(0, 1, 2))` : moyenne sur le lot (0), la hauteur (1) et la largeur (2). Il reste
  1 280 nombres : le poids de chaque carte.
- `conv_out[0] * poids` : chaque carte 7×7 est multipliée par son poids. `[0]` prend la première (et seule) image du lot.
- `reduce_sum(..., axis=-1)` : on additionne les 1 280 cartes pondérées. Il reste une carte 7×7.
- `tf.maximum(heatmap, 0)` : la ReLU, les valeurs négatives deviennent 0.
- `/ (tf.reduce_max(heatmap) + 1e-8)` : on divise par le maximum, la carte va de 0 à 1. Le `1e-8` évite une division
  par zéro si la carte est entièrement nulle.

### 5.4 Superposer la carte sur la radio

```python
h = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
h = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
return np.uint8(img_rgb * (1 - alpha) + h * alpha)
```

- `cv2.resize` : agrandit la carte 7×7 à la taille de la radio. Attention, OpenCV attend (largeur, hauteur), d'où
  `shape[1]` puis `shape[0]`. L'interpolation lisse les cases.
- `np.uint8(255 * h)` : passe de 0–1 à 0–255, le format attendu par OpenCV.
- `applyColorMap(..., COLORMAP_JET)` : 0 devient bleu foncé, 255 devient rouge.
- `cvtColor(..., COLOR_BGR2RGB)` : OpenCV range les couleurs dans l'ordre bleu-vert-rouge ; matplotlib attend
  rouge-vert-bleu. Sans cette ligne, le rouge et le bleu seraient inversés.
- `img_rgb * (1 - alpha) + h * alpha` : mélange à 60 % de radio et 40 % de couleur, pour voir la radio sous la carte.

### 5.5 Expliquer une radio

```python
img = charger_image(chemin)
x = PREPROCESS[backbone](img[None].copy())
prob = float(model.predict(x, verbose=0)[0, 0])
classe = int(prob >= 0.5)
heatmap = make_gradcam_heatmap(model, x, backbone, classe)
```

- `charger_image` : lit la radio exactement comme le pipeline (3 canaux, 224×224, interpolation bilinéaire).
- `img[None]` : ajoute la dimension du lot, (224, 224, 3) devient (1, 224, 224, 3). `.copy()` protège l'image
  d'origine, car certains prétraitements modifient le tableau sur place.
- `PREPROCESS[backbone]` : le prétraitement propre au modèle (étape 2).
- `classe = int(prob >= 0.5)` : on explique la décision réellement prise par le modèle.

## 6. Trois questions de jury

**Q1. Expliquez Grad-CAM en quelques phrases.**
On prend les cartes de la dernière couche convolutionnelle : 1 280 cartes 7×7 qui savent encore où se trouvent les
motifs dans l'image. On calcule le gradient du score de la classe par rapport à ces cartes : il dit à quel point chaque
zone fait varier la décision. La moyenne des gradients d'une carte donne son importance. On additionne les cartes
pondérées par leur importance, on garde la partie positive, on normalise entre 0 et 1, on agrandit à la taille de la
radio et on la superpose en couleurs. Rouge veut dire : cette zone a poussé le modèle vers sa réponse.

**Q2. Pourquoi avez-vous modifié le code du guide pour utiliser le logit ?**
Parce que le code du guide produisait des cartes vides. Il calcule le gradient de la probabilité, qui sort de la
sigmoïde. Quand le modèle est sûr à 100 %, la probabilité vaut exactement 1,0 en float32 et la sigmoïde ne varie plus :
son gradient est nul, donc la carte aussi. Nous l'avons vérifié : 0 case non nulle sur 49 pour la radio la plus
confiante de MobileNetV2, et 8 radios du test concernées pour VGG19. Le logit, le score avant la sigmoïde, ne sature
jamais. Quand la probabilité n'est pas saturée, les deux versions donnent la même carte, car seule l'échelle change.

**Q3. Les cartes montrent-elles que votre modèle est fiable ?**
Pas entièrement, et c'est justement l'intérêt de l'exercice. Sur les 624 radios du test, MobileNetV2 met en moyenne
46 % de sa carte au centre de l'image, qui ne fait que 36 % de la surface, et seulement 17 % sur les bords, qui en font
27 %. Il regarde donc surtout le thorax. Mais certaines cartes tombent sur le cou, le cœur ou l'abdomen plutôt que sur
les poumons, et VGG19 a, sur une fausse alerte, une zone rouge sur la lettre « R ». Grad-CAM reste une indication
grossière (7×7) et montre une corrélation, pas un raisonnement médical. Avant toute utilisation réelle, il faudrait
faire valider ces cartes par des radiologues.
