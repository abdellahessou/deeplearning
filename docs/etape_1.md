# Étape 1 – Données : téléchargement et nouveau split

## 1. Résumé

On a téléchargé le dataset Kaggle « Chest X-Ray Images (Pneumonia) » (5 856 radios) et corrigé son principal défaut :
le dossier `val` d'origine ne contient que **16 images**, beaucoup trop peu pour piloter un entraînement.
Le script `src/split_data.py` fusionne donc `train` et `val`, puis refait un découpage **stratifié 85 % / 15 %**
avec la seed 42, et recopie le tout dans `data/processed/`. Le dossier `test` d'origine est gardé **intact** :
il ne servira qu'à l'évaluation finale. Enfin, le notebook `notebooks/01_exploration.ipynb` explore les données
et produit deux figures pour le rapport.

## 2. Résultat obtenu

### Avant / après le split

| Split | Avant (Kaggle) | Après (`data/processed`) | NORMAL | PNEUMONIA | % PNEUMONIA |
|---|---|---|---|---|---|
| train | 5 216 | **4 447** | 1 147 | 3 300 | 74,2 % |
| val | 16 | **785** | 202 | 583 | 74,3 % |
| test | 624 | **624** (inchangé) | 234 | 390 | 62,5 % |
| Total | 5 856 | 5 856 | 1 583 | 4 273 | |

Sortie exacte du script :

```
> python src/split_data.py
train : 4447 images | NORMAL=1147  PNEUMONIA=3300
val   :  785 images | NORMAL= 202  PNEUMONIA= 583
test  :  624 images | NORMAL= 234  PNEUMONIA= 390
```

Ce sont exactement les chiffres annoncés par le guide.

### Vérifications effectuées

| Vérification | Résultat |
|---|---|
| Aucune image perdue : train + val = train + val d'origine | oui (1 349 NORMAL, 3 883 PNEUMONIA) |
| Stratification : part envoyée en validation | 14,97 % des NORMAL, 15,01 % des PNEUMONIA |
| Fichiers en commun entre train, val et test | 0 |
| Test copié identique à l'original | oui (mêmes noms ; contenu identique sur un échantillon de 100 images) |
| Reproductibilité (testée sur un mini-dataset factice) | deux lancements donnent exactement le même split |
| Notebook exécuté du début à la fin | 0 erreur |

### Fichiers produits

- `src/split_data.py` : le script de découpage.
- `notebooks/01_exploration.ipynb` : l'exploration, enregistrée avec ses sorties.
- `reports/figures/repartition_classes.png` : diagramme en barres des classes par split.
- `reports/figures/exemples_radios.png` : grille de 8 radios par classe.
- `data/raw/chest_xray/` (dataset d'origine, 1,2 Go) et `data/processed/` (nos splits, 1,15 Go) : non versionnés.

### Ce que montre l'exploration

- **Déséquilibre** : 74 % de pneumonies dans le train, soit environ 2,9 pneumonies pour 1 radio normale.
- **Le test est différent** : seulement 62,5 % de pneumonies (ratio 1,67).
- **Tailles très variées** : de 450×234 à 2619×2628 pixels sur un échantillon de 300 images (médiane 1272×888).
- **Format** : 286 images sur 300 en niveaux de gris (mode `L`), 14 en `RGB`.

### Limites à connaître (points forts devant un jury si on les annonce soi-même)

- **Un même patient peut être dans train et dans val.** Les pneumonies s'appellent `person<N>_bacteria_<k>.jpeg` :
  un patient a souvent plusieurs radios. Notre split est fait image par image, pas patient par patient.
  Résultat mesuré : 362 des 467 patients « pneumonie » du val sont aussi dans le train (468 images sur 583).
  Le score de validation sera donc un peu optimiste. Ce n'est pas grave pour son rôle (early stopping, choix du
  modèle), et cela ne touche pas le test.
- **Doublons dans le dataset** : 5 824 contenus distincts pour 5 856 fichiers. 9 images strictement identiques
  se retrouvent à la fois dans train et val. **Aucune** image identique entre le test et train/val.
- **Indépendance du test** : 155 identifiants `personN` existent à la fois dans train et dans test. Les noms de
  fichiers ne permettent pas de savoir s'il s'agit des mêmes patients ou d'une numérotation qui recommence ;
  les auteurs du dataset décrivent le test comme provenant de patients indépendants.

### Écarts par rapport au guide

- Sur ce PC, `kaggle.exe` est bloqué par le contrôle d'application de Windows : on utilise `python -m kaggle ...`.
- Le jeton Kaggle est dans `~/.kaggle/access_token` (nouveau format) au lieu de `kaggle.json`.
- `pandas<3` dans `requirements.txt` : pandas 3.1.1 est bloqué par la même protection Windows, pandas 2.3.3 fonctionne.
- `ipykernel` et `nbconvert` ajoutés à `requirements.txt` pour exécuter les notebooks.
- Le script vide `data/processed` avant de recopier, et s'arrête avec un message clair si le dataset est absent.

## 3. Notions théoriques

**Train / validation / test.** Trois jeux, trois rôles. Le *train* sert à apprendre (le modèle ajuste ses poids dessus).
La *validation* sert à piloter l'entraînement : quand s'arrêter, quel modèle garder. Le *test* sert à mesurer la
performance finale, une seule fois, sur des images jamais vues. Image simple : les exercices du cours, l'examen blanc,
l'examen final.

**Pourquoi 16 images de validation ne suffisent pas.** Avec 16 images, une seule erreur change l'accuracy de 6 points.
Le score de validation ferait des bonds au hasard, et l'early stopping s'arrêterait au mauvais moment.

**Stratification.** On garde la même proportion de classes dans chaque morceau. Sans elle, le hasard pourrait donner
une validation avec 80 % de pneumonies et un train avec 73 % : la validation ne représenterait plus le train.

**Déséquilibre des classes.** Quand une classe est beaucoup plus fréquente, l'accuracy devient trompeuse : un modèle
qui répond toujours « pneumonie » aurait 74 % d'accuracy sans avoir rien appris. D'où les poids de classes (étape 2)
et les métriques recall, F1 et AUC.

**Fuite de données (data leakage).** C'est quand une information du jeu d'évaluation « fuit » dans l'entraînement.
Le modèle a alors l'air meilleur qu'il ne l'est vraiment. La règle absolue : le test n'est jamais utilisé avant la fin.
La fuite « par patient » décrite plus haut en est une forme légère, limitée à la validation.

**Reproductibilité.** Deux ingrédients : la seed fixe (`random_state=SEED`) et une liste de fichiers toujours triée
dans le même ordre (`sorted`). Sans le tri, l'ordre des fichiers dépend du système, et la même seed donnerait un split différent.

## 4. Fonctions et classes

### Créées dans `src/split_data.py`

| Nom | Rôle | Paramètres |
|---|---|---|
| `lister_images(dossiers)` | Liste les images et leur étiquette (0 = NORMAL, 1 = PNEUMONIA) | `dossiers` : liste de splits d'origine, ex. `["train", "val"]` |
| `copier(chemins, labels, split)` | Copie chaque image dans `data/processed/<split>/<classe>/` | les chemins, leurs étiquettes, le nom du split |
| `main()` | Enchaîne : fusion, split stratifié, copie, affichage des effectifs | aucun |

### Utilisées

| Nom | Bibliothèque | Rôle | Paramètres importants |
|---|---|---|---|
| `train_test_split` | scikit-learn | Coupe des données en deux, au hasard | `test_size` (part du 2e morceau), `stratify` (étiquettes à respecter), `random_state` (seed) |
| `Path.iterdir()` | pathlib | Parcourt le contenu d'un dossier | aucun |
| `Path.suffix` | pathlib | Extension du fichier (`.jpeg`) | aucun |
| `Path.mkdir` | pathlib | Crée un dossier | `parents=True` (crée les dossiers parents), `exist_ok=True` (pas d'erreur s'il existe) |
| `shutil.copy` | shutil | Copie un fichier | source, destination |
| `shutil.rmtree` | shutil | Supprime un dossier et son contenu | le dossier |
| `enumerate` | Python | Donne l'indice et la valeur dans une boucle | la liste |
| `zip` | Python | Parcourt deux listes en parallèle | les listes |
| `ax.bar`, `ax.bar_label` | Matplotlib | Trace des barres, écrit la valeur au-dessus | positions, hauteurs, largeur, couleur |
| `np.random.default_rng(SEED)` | NumPy | Générateur aléatoire reproductible | la seed |
| `Image.open` | Pillow | Ouvre une image (taille, mode, pixels) | le chemin |

## 5. Le code important, expliqué ligne par ligne

### 5.1 Lister les images et leurs étiquettes

```python
EXTENSIONS = {".jpeg", ".jpg", ".png"}

def lister_images(dossiers):
    chemins, labels = [], []
    for d in dossiers:
        for label, classe in enumerate(CLASSES):
            for p in sorted((RAW_DIR / d / classe).iterdir()):
                if p.suffix.lower() in EXTENSIONS:
                    chemins.append(p)
                    labels.append(label)
    return chemins, labels
```

- `EXTENSIONS` : les types de fichiers acceptés. Le dataset n'a que des `.jpeg`, mais on ignore ainsi tout fichier parasite.
- `chemins, labels = [], []` : deux listes vides, remplies en parallèle. L'image n° i a l'étiquette n° i.
- `for d in dossiers` : on parcourt les splits demandés, par exemple `train` puis `val`. C'est ici que se fait la **fusion**.
- `for label, classe in enumerate(CLASSES)` : `enumerate` donne `(0, "NORMAL")` puis `(1, "PNEUMONIA")`.
  L'étiquette vient donc de la position dans la liste `CLASSES` de `config.py`.
- `sorted((RAW_DIR / d / classe).iterdir())` : tout le contenu du dossier, **trié** pour avoir toujours le même ordre.
- `if p.suffix.lower() in EXTENSIONS` : on ne garde que les images.
- `append` : on ajoute le chemin et son étiquette. `return` renvoie les deux listes.

### 5.2 Copier les images

```python
def copier(chemins, labels, split):
    for p, y in zip(chemins, labels):
        dest = DATA_DIR / split / CLASSES[y]
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dest / p.name)
```

- `zip(chemins, labels)` : parcourt les deux listes ensemble, `p` est un chemin et `y` son étiquette (0 ou 1).
- `dest = DATA_DIR / split / CLASSES[y]` : le dossier d'arrivée, par exemple `data/processed/val/PNEUMONIA`.
  `CLASSES[y]` retransforme l'étiquette 1 en nom de dossier.
- `dest.mkdir(...)` : crée ce dossier s'il n'existe pas encore.
- `shutil.copy(p, dest / p.name)` : copie l'image en gardant son nom. On **copie**, on ne déplace pas : le dataset d'origine reste intact.

Cette structure « un dossier par classe » est celle que Keras sait lire directement à l'étape 2.

### 5.3 Le split stratifié

```python
def main():
    if not (RAW_DIR / "train").exists():
        raise SystemExit(f"Dataset introuvable dans {RAW_DIR} : ...")

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    X, y = lister_images(["train", "val"])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=SEED)

    X_te, y_te = lister_images(["test"])
```

- Le premier `if` : si le dataset n'est pas téléchargé, on s'arrête avec un message clair plutôt qu'une erreur incompréhensible.
- Le second `if` : on efface l'ancien `data/processed`. Sinon, en relançant avec une autre seed, des images de
  l'ancien split resteraient mélangées au nouveau, et une même image pourrait se retrouver dans train **et** val.
- `X, y = lister_images(["train", "val"])` : la fusion. `X` contient 5 232 chemins, `y` leurs étiquettes.
- `train_test_split(...)` : la ligne centrale de l'étape.
  - `test_size=VAL_SIZE` : 15 % partent en validation, 85 % restent en train.
  - `stratify=y` : le découpage respecte la proportion des classes de `y`.
  - `random_state=SEED` : le tirage est toujours le même.
  - Elle renvoie 4 listes : chemins du train, chemins du val, étiquettes du train, étiquettes du val.
- `lister_images(["test"])` : le test est simplement listé, sans aucun tirage. Il reste tel quel.

### 5.4 Copie et affichage

```python
    splits = {"train": (X_tr, y_tr), "val": (X_val, y_val), "test": (X_te, y_te)}
    for nom, (chemins, labels) in splits.items():
        copier(chemins, labels, nom)
        print(f"{nom:5s} : {len(chemins):4d} images | "
              f"NORMAL={labels.count(0):4d}  PNEUMONIA={labels.count(1):4d}")

if __name__ == "__main__":
    main()
```

- `splits` : un dictionnaire qui associe à chaque nom de split ses chemins et ses étiquettes.
- La boucle copie chaque split puis affiche ses effectifs. `labels.count(0)` compte les étiquettes 0, donc les NORMAL.
- `{nom:5s}` et `{...:4d}` : mise en forme pour aligner les colonnes à l'affichage.
- `if __name__ == "__main__":` : `main()` ne s'exécute que si on lance le fichier directement
  (`python src/split_data.py`), pas si un autre script l'importe.

### 5.5 Le notebook : compter et tracer

```python
comptes = pd.DataFrame(
    {c: [len(list((DATA_DIR / s / c).glob("*"))) for s in SPLITS] for c in CLASSES},
    index=SPLITS)
comptes["Total"] = comptes.sum(axis=1)
comptes["% PNEUMONIA"] = (100 * comptes["PNEUMONIA"] / comptes["Total"]).round(1)
```

- On construit un tableau pandas : une ligne par split, une colonne par classe. Chaque case est le nombre de
  fichiers du dossier correspondant.
- `sum(axis=1)` additionne les colonnes ligne par ligne pour obtenir le total.
- La dernière ligne calcule le pourcentage de pneumonies : c'est elle qui révèle le déséquilibre.

```python
rng = np.random.default_rng(SEED)
choisis = rng.choice(len(fichiers), size=8, replace=False)
```

- `default_rng(SEED)` crée un générateur aléatoire reproductible : la grille montre toujours les mêmes radios.
- `rng.choice(..., size=8, replace=False)` tire 8 numéros d'images différents (`replace=False` = pas deux fois la même).

## 6. Trois questions de jury

**Q1. Pourquoi avoir refait le split au lieu d'utiliser celui de Kaggle ?**
Le dossier de validation d'origine ne contient que 16 images (8 par classe). Une seule erreur y change l'accuracy
de 6 points : l'early stopping et le choix du meilleur modèle auraient reposé sur du bruit. En fusionnant train et val
puis en redécoupant à 85/15, on obtient 785 images de validation, avec la même proportion de classes que le train
grâce à la stratification (74,2 % et 74,3 % de pneumonies). Le test, lui, n'a pas été modifié.

**Q2. Votre découpage garantit-il qu'il n'y a aucune fuite de données ?**
Entre le test et le reste, oui au niveau des images : aucun fichier en commun et aucune image identique, et le test
n'est jamais utilisé avant l'évaluation finale. Entre train et validation, non, pas complètement : le split est fait
image par image, alors qu'un patient a souvent plusieurs radios. 362 des 467 patients « pneumonie » de la validation
ont aussi des radios dans le train. Le score de validation est donc un peu optimiste. Une amélioration possible
serait un split par patient (`GroupShuffleSplit` de scikit-learn). Nous avons gardé le protocole simple, car la
validation ne sert qu'à piloter l'entraînement : les chiffres annoncés viennent uniquement du test.

**Q3. Le dataset est déséquilibré : quelles conséquences, et le test a-t-il la même répartition ?**
Le train contient 74 % de pneumonies. Un modèle qui répondrait toujours « pneumonie » aurait 74 % d'accuracy sans
rien avoir appris : l'accuracy seule est trompeuse. On compense avec des poids de classes dans la loss (étape 2) et on
juge le modèle avec le recall, le F1 et l'AUC. Le test, lui, ne contient que 62,5 % de pneumonies : sa répartition
est différente de celle du train. C'est une des raisons pour lesquelles le score sur le test peut être plus bas que
celui de validation.
