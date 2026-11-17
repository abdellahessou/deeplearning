# Étape 8 – Rapport, README et soutenance

## 1. Résumé

On a transformé le travail des étapes 0 à 7 en livrables. Le **README** (8a) présente maintenant le projet complet :
objectif, résultats, installation, données, une commande par script, l'application et la structure du dépôt.
Le **brouillon de rapport** `reports/rapport.md` (8b) suit les 10 parties du plan du guide et ne cite que les chiffres
réellement obtenus ; chaque partie signale les points faibles qu'un jury pourrait relever. Le **plan de soutenance**
`reports/plan_soutenance.md` (8c) propose 11 diapositives pour 15 minutes, avec la figure et le texte à dire pour
chacune. Ce sont des **brouillons** : tu dois les relire, les reformuler et les compléter.

## 2. Résultat obtenu

### Fichiers produits ou modifiés

| Fichier | Contenu | Taille |
|---|---|---|
| `README.md` | Objectif, tableau de résultats, 3 figures, installation, versions testées, données, toutes les commandes, application, structure | environ 1 350 mots |
| `reports/rapport.md` | Brouillon de rapport en 10 parties, 11 figures, références, annexe des points faibles | environ 6 300 mots |
| `reports/plan_soutenance.md` | 11 diapositives minutées, check-list du jour J, questions à préparer | environ 2 000 mots |
| `requirements.txt` | Vérifié, inchangé | 12 dépendances |

### Vérification de `requirements.txt`

Tous les modules importés par le code (`src/`, `app/`) sont couverts : `tensorflow`, `numpy`, `pandas`, `sklearn`
(scikit-learn), `matplotlib`, `seaborn`, `PIL` (pillow), `cv2` (opencv-python-headless), `streamlit`. S'y ajoutent
`kaggle` (téléchargement), `ipykernel` et `nbconvert` (notebooks). Les autres imports (`argparse`, `pathlib`, `shutil`,
`sys`, `time`) font partie de Python. Les versions testées sont listées dans le README ; elles ne sont pas imposées
dans `requirements.txt`, pour que le même fichier fonctionne aussi sur Colab, qui a ses propres versions.

### Le rapport

- **Plan** : les 10 parties du guide (introduction, état de l'art, données, méthode, résultats, explicabilité,
  application, discussion, éthique, conclusion), plus un résumé et une annexe.
- **Chiffres** : tous viennent des fichiers du projet. Les informations tirées de la littérature (auteurs, dates,
  chiffres de l'OMS, nom de l'hôpital, lois) sont marquées **[à vérifier]**.
- **Points faibles** : 10 encadrés dans le texte, et un tableau récapitulatif en annexe avec, pour chacun, la section
  où il est traité et une réponse courte.
- **Figures** : 11 figures, liées par des chemins relatifs (`figures/...`) ; tous les liens ont été vérifiés.
- **Longueur** : environ 6 300 mots, soit à peu près 15 à 18 pages une fois mis en page avec les figures. Il faudra
  l'enrichir pour atteindre 20 à 30 pages (voir « À faire de ton côté »).

### La soutenance

| Partie | Diapositives | Durée |
|---|---|---|
| Contexte et données | 1 à 3 | 4 min |
| Méthode et entraînement | 4 et 5 | 2 min 30 |
| Résultats | 6 à 8 | 4 min 15 |
| Explicabilité et démonstration | 9 et 10 | 3 min 15 |
| Limites et conclusion | 11 | 1 min |

La démonstration (diapositive 10) reprend les quatre cas testés à l'étape 7, dont le Grad-CAM de VGG19 sur la
lettre « R ».

### Écarts par rapport au guide

- **Plan de 11 diapositives** (le guide en demande 10 à 12), avec en plus un minutage et une check-list.
- **Annexe des points faibles** dans le rapport, pour préparer les questions.
- **Captures d'écran non incluses** : elles doivent être prises par toi. Leur emplacement est prévu dans le README
  (en commentaire, pour ne pas afficher d'images cassées) et dans le rapport (section 7.2).

### À faire de ton côté

1. **Prendre les captures d'écran** de l'application et les enregistrer dans `reports/figures/` sous les noms
   `app_pneumonie.png`, `app_normal.png`, `app_seuil_1.png` et `app_seuil.png`. Dans le README, il suffit ensuite d'enlever les balises
   `<!--` et `-->` autour des trois lignes d'images.
2. **Vérifier les références** marquées [à vérifier] dans le rapport, et les citer selon le format demandé.
3. **Relire et reformuler le rapport** avec tes mots ; l'enrichir pour atteindre 20 à 30 pages, par exemple :
   - expliquer plus en détail les architectures (schémas des blocs résiduels et des blocs de MobileNetV2) ;
   - ajouter les courbes de ResNet50 et VGG19, et les matrices de confusion des trois modèles ;
   - décrire le déroulement du projet (environnement, problèmes rencontrés et solutions).
4. **Mettre le rapport en page** (Word, Google Docs ou LaTeX) et **créer la présentation** à partir du plan.
5. **Répéter la soutenance** à voix haute, chronomètre en main.
6. **Prévoir les modèles** : les fichiers `.keras` ne sont pas sur GitHub. Garde une copie (clé USB ou Drive) pour
   l'ordinateur de la soutenance.

## 3. Notions théoriques

**Reproductibilité.** Quelqu'un d'autre doit pouvoir refaire exactement nos résultats. Il lui faut : le code (dépôt
Git), les dépendances (`requirements.txt` et versions testées), les données (commande de téléchargement), l'ordre des
commandes (README) et une seed fixe. C'est ce que vérifie un jury quand il demande « comment je relance votre
projet ? ».

**Fiche expérimentale.** Le résumé chiffré d'une expérience : données, protocole, hyperparamètres, résultats et
figures. Dans notre projet, elle correspond au tableau de comparaison, aux courbes, aux matrices de confusion et aux
courbes ROC, rassemblés dans le README et la partie 5 du rapport.

**Honnêteté des résultats.** Un rapport scientifique annonce les chiffres qui reflètent la réalité (le test, pas la
validation), montre les erreurs et discute les limites. Un jury fait plus confiance à un étudiant qui annonce
lui-même ses points faibles qu'à un étudiant qui les cache.

**Significativité.** Avec un test de 624 images, un écart d'un point d'accuracy correspond à environ 6 images :
c'est moins que l'incertitude de la mesure (environ ± 3 points). On ne peut pas dire qu'un modèle est meilleur
qu'un autre sur un tel écart.

**Rédiger une soutenance.** Une idée par diapositive, une figure qui la porte, et un message oral qui l'explique.
Le texte de la diapositive résume, l'oral raconte.

## 4. Fonctions et classes

Cette étape ne crée pas de code. Voici les outils et fichiers utilisés pour produire les livrables :

| Nom | Rôle | Utilisé pour |
|---|---|---|
| `reports/comparaison.csv` | Résultats des 3 modèles sur le test (étape 4) | Tableaux du README et du rapport |
| `reports/rapport_<modèle>.txt` | Rapport de classification (étape 5) | Section 5.3 du rapport |
| `reports/seuils_<modèle>.csv` | Effet du seuil (étape 5) | Section 5.4 du rapport, diapositive 8 |
| `models/<modèle>_phase<N>_history.csv` | Historiques d'entraînement (étape 4) | Section 5.1 du rapport |
| `reports/figures/*.png` | Toutes les figures | Rapport, README, diapositives |
| `docs/etape_0.md` à `etape_7.md` | Fiches détaillées de chaque étape | Source du texte et des questions de jury |
| Markdown `![texte](chemin)` | Insère une image dans un fichier `.md` | Figures du README et du rapport |
| Commentaire HTML `<!-- ... -->` | Masque du texte sans le supprimer | Emplacement des futures captures dans le README |

## 5. Le code important, expliqué ligne par ligne

Le « code » de cette étape, c'est la suite de commandes du README qui permet de refaire tout le projet.

### 5.1 Installer l'environnement

```bash
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install -r requirements.txt
```

- `py -3.11 -m venv .venv311` : crée un environnement Python 3.11 isolé dans le dossier `.venv311`.
- `activate` : l'active ; les commandes `python` et `pip` utilisent alors cet environnement.
- `pip install -r requirements.txt` : installe toutes les bibliothèques listées.

### 5.2 Préparer les données

```bash
python -m kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw --unzip
python src/split_data.py
```

- La première commande télécharge et décompresse le dataset dans `data/raw/` (il faut un jeton Kaggle).
- La seconde refait le découpage stratifié (seed 42) dans `data/processed/`.

### 5.3 Entraîner, comparer, évaluer, expliquer

```bash
python src/train.py --model mobilenetv2          # sur Colab de préférence
python src/compare.py
python src/evaluate.py --model mobilenetv2
python src/gradcam.py --model mobilenetv2
```

- `train.py` : phase 1 puis fine-tuning, sauvegarde dans `models/` (à faire pour chacun des 3 modèles).
- `compare.py` : tableau et graphique de comparaison sur le test.
- `evaluate.py` : rapport, matrice de confusion, courbes ROC et précision-rappel, seuils, erreurs.
- `gradcam.py` : cartes de chaleur.

### 5.4 Lancer l'application

```bash
python -m streamlit run app/app.py
```

- Ouvre l'application sur http://localhost:8501.

### 5.5 Afficher une figure dans un fichier Markdown

```markdown
![Matrice de confusion](reports/figures/cm_mobilenetv2.png)
```

- `!` indique une image ; le texte entre crochets s'affiche si l'image manque ; le chemin entre parenthèses est
  relatif au fichier `.md`. C'est pourquoi le README écrit `reports/figures/...` et le rapport, qui est déjà dans
  `reports/`, écrit `figures/...`.

## 6. Trois questions de jury

**Q1. Comment peut-on refaire votre projet de zéro ?**
Tout est décrit dans le README : créer l'environnement Python 3.11 et installer `requirements.txt`, télécharger le
dataset avec la commande Kaggle, lancer `split_data.py` pour refaire le découpage avec la seed 42, entraîner les trois
modèles avec `train.py` (sur Colab avec le notebook fourni, faute de GPU), puis lancer `compare.py`, `evaluate.py`,
`gradcam.py` et l'application. Le code est sur GitHub et les versions testées sont indiquées. Seuls les fichiers des
modèles entraînés ne sont pas dans le dépôt, car ils sont trop lourds.

**Q2. Quel est, selon vous, le principal point faible de votre travail ?**
L'évaluation pendant l'entraînement : notre validation a été découpée image par image, et 362 de ses patients
atteints de pneumonie ont aussi des radios dans le train. Elle annonçait 96 % d'accuracy alors que le test donne 84 %.
Cela n'a pas faussé le résultat final, mesuré sur un test séparé, mais cela nous a empêchés de choisir un seuil
fiable. Si c'était à refaire, je découperais par patient, et je choisirais le seuil sur cette validation propre.

**Q3. Votre outil pourrait-il être utilisé dans un hôpital ?**
Non, pas en l'état. Il a été entraîné sur des radios d'enfants d'un seul hôpital, sa spécificité sur le test n'est
que de 60 %, et il classe même une image qui n'est pas une radio en pneumonie à 100 %. Il faudrait le valider sur des
données de l'hôpital concerné, probablement le réentraîner, faire valider ses décisions et ses cartes Grad-CAM par
des radiologues, et respecter la réglementation sur les données de santé. C'est un prototype pédagogique, et
l'application le dit clairement.
