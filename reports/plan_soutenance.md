# Plan de la soutenance (15 minutes, 11 diapositives)

*Brouillon à adapter. Le texte « À dire » est une proposition : reformule-le avec tes mots et entraîne-toi à voix
haute, chronomètre en main. Toutes les figures sont dans `reports/figures/`.*

| N° | Diapositive | Durée | Cumul |
|---|---|---|---|
| 1 | Titre et problématique | 1 min | 1 min |
| 2 | Les données | 1 min 30 | 2 min 30 |
| 3 | Préparation des données | 1 min 30 | 4 min |
| 4 | Méthode : transfer learning en deux phases | 1 min 30 | 5 min 30 |
| 5 | Entraînement | 1 min | 6 min 30 |
| 6 | Comparaison des 3 modèles | 1 min 30 | 8 min |
| 7 | Évaluation du modèle retenu | 1 min 30 | 9 min 30 |
| 8 | Effet du seuil et validation vs test | 1 min 15 | 10 min 45 |
| 9 | Explicabilité : Grad-CAM | 1 min 30 | 12 min 15 |
| 10 | Démonstration de l'application | 1 min 45 | 14 min |
| 11 | Limites, éthique et perspectives | 1 min | 15 min |

---

## 1. Détection de pneumonie sur radiographies thoraciques

**Points clés**
- La pneumonie : une cause majeure de décès chez l'enfant ; diagnostic par radiographie.
- Objectif : une aide qui signale les pneumonies, **sans en rater** (recall prioritaire).
- Problématique : un CNN pré-entraîné sur des photos peut-il apprendre avec ~4 400 radios, et peut-on comprendre ses décisions ?

**Figure** : deux radios côte à côte, une normale et une pneumonie (extraites de `exemples_radios.png`).

**À dire** : « Une radio normale a des poumons sombres, remplis d'air ; une pneumonie apparaît comme une zone plus
blanche. Lire ces radios demande un radiologue, qui n'est pas toujours disponible. Notre objectif est un outil qui
signale les radios suspectes. Le critère principal est le recall : rater un enfant malade est bien plus grave qu'une
fausse alerte, qui sera vérifiée par un médecin. »

---

## 2. Les données : 5 856 radios d'enfants

**Points clés**
- Dataset Kaggle de Kermany et al. : radios de face d'enfants, deux classes.
- **Déséquilibre** : 74 % de pneumonies dans le train.
- Le dossier de validation d'origine n'a que **16 images**.

**Figure** : `repartition_classes.png`.

**À dire** : « Il y a presque trois pneumonies pour une radio normale. Un modèle qui dirait toujours "pneumonie"
aurait déjà 74 % d'accuracy : c'est pourquoi l'accuracy seule ne suffit pas. Deuxième problème : la validation
fournie n'a que 16 images. Une seule erreur y fait varier l'accuracy de 6 points, impossible de piloter un
entraînement avec ça. »

---

## 3. Préparation des données

**Points clés**
- Nouveau split **stratifié 85 / 15**, seed 42 : 4 447 train, 785 validation ; **test de 624 images intact**.
- Poids de classes **1,94 / 0,67** ; augmentation réaliste (rotation ±10°, zoom, translation, contraste, **pas de flip**).
- Pipeline identique pour tous : RGB, 224×224, `preprocess_input` propre à chaque modèle.

**Figure** : `augmentation.png`.

**À dire** : « On a fusionné train et val et refait un découpage stratifié, en gardant le test intact pour
l'évaluation finale. Pour le déséquilibre, une erreur sur une radio normale compte 2,9 fois plus dans la loss.
L'augmentation crée des variantes réalistes : patient penché, mal centré. Pas de symétrie, car le cœur est à gauche,
une radio inversée n'existe pas. Limite que j'ai mesurée : le split est fait par image, et 362 patients de la
validation ont aussi des radios dans le train. J'y reviendrai. »

---

## 4. Méthode : transfer learning en deux phases

**Points clés**
- 3 backbones ImageNet : **MobileNetV2** (2,3 M paramètres), **ResNet50** (23,6 M), **VGG19** (20 M).
- Tête : Global Average Pooling → Dropout 0,3 → Dense sigmoïde.
- Phase 1 : base gelée, lr 0,001. Phase 2 : **dernier bloc dégelé**, lr 0,00001, **BatchNorm gelées**.

**Figure** : un schéma simple : radio → backbone (gelé / dernier bloc dégelé) → GAP → Dropout → sortie.
*(À dessiner dans le logiciel de présentation.)*

**À dire** : « Avec 4 400 images, un CNN partant de zéro surapprendrait. On réutilise un réseau déjà entraîné sur
1,2 million de photos : ses premières couches détectent des bords et des textures, utiles aussi sur des radios.
On entraîne d'abord seulement la tête, puis on réajuste le dernier bloc avec un learning rate cent fois plus petit,
pour retoucher les poids sans les détruire. Le guide dégelait 30 couches partout, mais VGG19 n'en a que 22 : on a
dégelé le dernier bloc de chaque modèle pour que la comparaison soit juste. »

---

## 5. Entraînement

**Points clés**
- Entraînement sur **Google Colab (GPU T4)**, le PC n'ayant pas de GPU utilisable.
- Le fine-tuning améliore la loss de validation des 3 modèles (VGG19 : 0,131 → 0,084).
- Début de surapprentissage après le fine-tuning, stoppé par l'**early stopping**.

**Figure** : `courbes_resnet50.png` (le surapprentissage y est le plus visible).

**À dire** : « Le trait vertical marque le début du fine-tuning. La loss d'entraînement chute, mais la loss de
validation oscille : le modèle commence à apprendre par cœur. L'early stopping a arrêté chaque phase 2 après 6 ou
7 epochs et a restauré les poids de la meilleure epoch. »

---

## 6. Comparaison des 3 modèles sur le test

**Points clés**
- Accuracy **83,2 à 84,3 %**, recall **97,4 à 99,0 %**, AUC **0,950 à 0,968** : à égalité (7 images d'écart sur 624).
- MobileNetV2 : **10 fois moins de paramètres**, ≈ 10 ms par image contre 29 et 92 ms.
- **Modèle retenu : MobileNetV2.**

**Figure** : le tableau de comparaison (à recopier depuis `comparaison.csv`) et `comparaison_modeles.png`.

**À dire** : « Sur 624 radios jamais vues, les trois modèles se valent : l'écart est plus petit que la marge
d'incertitude, environ 3 points. À performance égale, on prend le plus simple : MobileNetV2 est dix fois plus léger
et trois à neuf fois plus rapide, idéal pour une application. Un gros modèle n'apporte rien ici avec si peu
d'images. »

---

## 7. Évaluation du modèle retenu

**Points clés**
- **5 pneumonies ratées sur 390** : recall 98,7 %.
- **93 fausses alertes sur 234 radios normales** : spécificité 60 %.
- AUC **0,962** : le modèle classe bien, mais le seuil de 0,5 coupe mal.

**Figure** : `cm_mobilenetv2.png` et `roc_mobilenetv2.png`.

**À dire** : « La matrice de confusion montre des erreurs très déséquilibrées : presque aucune pneumonie ratée, mais
quatre radios normales sur dix déclarées malades. Quand le modèle dit "normal", il a raison à 96,6 %. L'AUC de 0,96
signifie que dans 96 % des cas, il donne une probabilité plus haute à la radio malade qu'à la saine. Le problème vient
donc surtout du seuil, pas du classement. »

---

## 8. Effet du seuil et écart validation / test

**Points clés**
- Seuil 0,5 → 0,7 : **25 fausses alertes en moins, 3 pneumonies ratées en plus** ; c'est un compromis.
- On ne choisit **pas** le seuil sur le test (résultat optimiste) : il reste à 0,5, réglable dans l'application.
- **96 % en validation, 84 % au test** : validation contaminée par des patients du train, test différent.

**Figure** : le tableau des seuils (`seuils_mobilenetv2.csv`), avec la ligne 0,5 en gras.

**À dire** : « Monter le seuil réduit les fausses alertes mais fait rater des malades. Ce choix est médical, pas
seulement technique. Et surtout, je ne l'ai pas optimisé sur le test, sinon le résultat serait trop beau. Point
important : en validation, on avait 96 %. La validation partage des patients avec le train, elle est trop facile.
Le chiffre honnête, c'est celui du test : 84 %. »

---

## 9. Explicabilité : Grad-CAM

**Points clés**
- Grad-CAM : carte des zones qui ont le plus pesé, à partir des gradients de la dernière couche convolutionnelle.
- MobileNetV2 regarde surtout le thorax : **46 % de la carte au centre** (36 % de la surface), 17 % sur les bords.
- Mais **VGG19 s'appuie parfois sur la lettre « R »** : biais « Clever Hans ».

**Figure** : `gradcam_mobilenetv2.png`, puis la colonne « faux positif » de `gradcam_vgg19.png`.

**À dire** : « Pour savoir sur quoi le modèle se base, on calcule le gradient du score par rapport aux cartes de la
dernière couche : il dit quelles zones font varier la décision. En rouge, les zones déterminantes. Sur toutes les
radios du test, MobileNetV2 se concentre sur le thorax. Mais VGG19, sur une fausse alerte, regarde la lettre "R"
imprimée sur la radio : un indice sans rapport avec la maladie. Petite correction technique : le code de référence
donnait des cartes vides quand le modèle était sûr à 100 %, on calcule donc le gradient avant la sigmoïde. »

---

## 10. Démonstration de l'application

**Points clés**
- Streamlit : choix du modèle et du seuil, upload d'une radio, Grad-CAM, probabilité.
- Même prétraitement qu'à l'entraînement : **probabilités identiques** à l'évaluation.
- Avertissement : **ne remplace pas un diagnostic médical**.

**Figure** : démonstration en direct (prévoir les captures d'écran en secours si la démo échoue).

**À dire / à faire** (lancer l'application **avant** la soutenance : le premier chargement prend 15 secondes) :
1. MobileNetV2, exemple `PNEUMONIA – person100_bacteria_475.jpeg` : « Pneumonie, 72,8 %. À droite, les zones
   rouges montrent ce qui a fait pencher le modèle. »
2. Exemple `NORMAL – IM-0001-0001.jpeg` : « Normal, 2,5 %. »
3. VGG19, même radio, seuil 0,10 : « Ici VGG19 hésite, 53,8 %, et regardez : la carte est sur la lettre "R". »
4. Remonter le seuil à 0,60 : « La prédiction devient normale : c'est le seuil qui décide. »

---

## 11. Limites, éthique et perspectives

**Points clés**
- Limites : un seul hôpital pédiatrique, spécificité de 60 %, fuite par patient en validation, image non radiologique
  classée pneumonie à 100 %.
- Éthique : outil expérimental, décision au médecin, données de santé traitées en local.
- Perspectives : split par patient, seuil choisi sur une validation propre, données multi-centres, classes
  bactérienne / virale, version mobile (TensorFlow Lite).

**Figure** : aucune, ou une liste visuelle à trois colonnes (limites / éthique / perspectives).

**À dire** : « Ce modèle ne doit pas être utilisé sur des patients : il a été entraîné sur des enfants d'un seul
hôpital, et il répond même à une image qui n'est pas une radio. Ce que j'en retiens : le chiffre affiché dépend
beaucoup de la façon d'évaluer, et l'explicabilité permet de repérer des comportements qu'aucune métrique ne montre.
Je vous remercie. »

---

## Préparer les questions

Les 3 questions de jury de chaque fiche `docs/etape_N.md` et l'annexe du rapport (`reports/rapport.md`) couvrent les
questions les plus probables. À relire en priorité :

- Pourquoi le transfer learning ? (`docs/etape_3.md`, Q1)
- Pourquoi le recall plutôt que l'accuracy ? (`docs/etape_5.md`, Q1)
- Pourquoi 96 % en validation et 84 % au test ? (`docs/etape_4.md`, Q1)
- Pourquoi geler les BatchNorm ? (`docs/etape_4.md`, Q2)
- Pourquoi ne pas monter le seuil ? (`docs/etape_5.md`, Q2)
- Comment marche Grad-CAM, et pourquoi le logit ? (`docs/etape_6.md`, Q1 et Q2)
- Le modèle marcherait-il dans un hôpital marocain ? (`reports/rapport.md`, section 8.2)

## Check-list du jour J

- [ ] Application lancée et testée sur l'ordinateur de présentation (modèles `.keras` présents dans `models/`).
- [ ] Captures d'écran de secours dans la présentation.
- [ ] Présentation chronométrée sous 15 minutes.
- [ ] Rapport relu, références vérifiées, captures d'écran insérées.
