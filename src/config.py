"""Constantes du projet, regroupées ici pour que tous les scripts utilisent les mêmes valeurs."""
from pathlib import Path

# --- Chemins ---
ROOT = Path(__file__).resolve().parents[1]       # dossier racine du projet
RAW_DIR = ROOT / "data" / "raw" / "chest_xray"   # dataset Kaggle d'origine
DATA_DIR = ROOT / "data" / "processed"           # nos splits train/val/test
MODELS_DIR = ROOT / "models"                     # modèles .keras sauvegardés
FIG_DIR = ROOT / "reports" / "figures"           # figures du rapport

# --- Paramètres communs à tous les scripts ---
IMG_SIZE = (224, 224)              # taille attendue par les modèles ImageNet
BATCH_SIZE = 32                    # nombre d'images traitées ensemble
SEED = 42                          # graine fixe = résultats reproductibles
CLASSES = ["NORMAL", "PNEUMONIA"]  # 0 = NORMAL, 1 = PNEUMONIA
VAL_SIZE = 0.15                    # 15 % du train sert à la validation
