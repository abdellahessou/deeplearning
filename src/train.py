"""Étapes 3 et 4 : entraînement d'un modèle en deux phases.

Phase 1 : base gelée, seule la tête apprend.   Phase 2 : fine-tuning des dernières couches.

Exemples :
    python src/train.py --model mobilenetv2 --epochs2 0     # étape 3 : phase 1 seule
    python src/train.py --model resnet50                    # phases 1 et 2
"""
import argparse

import matplotlib
matplotlib.use("Agg")               # on enregistre les figures sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tensorflow import keras

from config import MODELS_DIR, FIG_DIR, SEED
from data import get_datasets, class_weights
from models import build_model, compile_model, fine_tune, N_UNFREEZE

keras.utils.set_random_seed(SEED)   # fixe les tirages de Python, NumPy et TensorFlow


def callbacks(name, phase):
    """Les 4 « surveillants » de l'entraînement, appelés à la fin de chaque epoch."""
    return [
        # Arrête si la val_loss ne s'améliore plus pendant 5 epochs, et reprend les meilleurs poids
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        # Sauvegarde le meilleur modèle de la phase
        keras.callbacks.ModelCheckpoint(str(MODELS_DIR / f"{name}_{phase}.keras"),
                                        monitor="val_loss", save_best_only=True),
        # Divise le learning rate par 5 si la val_loss stagne 2 epochs
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
        # Enregistre l'historique pour les courbes
        keras.callbacks.CSVLogger(str(MODELS_DIR / f"{name}_{phase}_history.csv")),
    ]


def tracer_courbes(name):
    """Trace la loss et l'accuracy train/val à partir des CSV d'historique."""
    fichiers = [MODELS_DIR / f"{name}_{phase}_history.csv" for phase in ("phase1", "phase2")]
    historiques = [pd.read_csv(f) for f in fichiers if f.exists()]
    h = pd.concat(historiques, ignore_index=True)
    epochs = range(1, len(h) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metrique, titre in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        ax.plot(epochs, h[metrique], color="#2a78d6", linewidth=2, marker="o", label="train")
        ax.plot(epochs, h[f"val_{metrique}"], color="#eb6834", linewidth=2, marker="o", label="validation")
        if len(historiques) == 2:   # trait vertical au début du fine-tuning
            ax.axvline(len(historiques[0]) + 0.5, color="gray", linestyle="--", label="début du fine-tuning")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # graduations entières : 1, 2, 3...
        ax.set_title(f"{titre} – {name}")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"courbes_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Courbes sauvegardées :", FIG_DIR / f"courbes_{name}.png")


def entrainer(name, epochs1=15, epochs2=10):
    MODELS_DIR.mkdir(exist_ok=True)
    train, val, _ = get_datasets(name)      # le test n'est PAS utilisé pendant l'entraînement
    cw = class_weights()
    model, base = build_model(name)

    # On repart de zéro : sinon un ancien historique de phase 2 se mélangerait aux nouvelles courbes
    for phase in ("phase1", "phase2"):
        (MODELS_DIR / f"{name}_{phase}_history.csv").unlink(missing_ok=True)

    # Phase 1 : backbone gelé, seule la tête apprend (lr élevé)
    compile_model(model, lr=1e-3)
    model.fit(train, validation_data=val, epochs=epochs1,
              class_weight=cw, callbacks=callbacks(name, "phase1"))

    # Phase 2 : fine-tuning des dernières couches (lr très faible)
    if epochs2 > 0:
        fine_tune(base, N_UNFREEZE[name])   # dégèle le dernier bloc, BatchNorm toujours gelées
        compile_model(model, lr=1e-5)       # recompiler après avoir changé trainable
        model.fit(train, validation_data=val, epochs=epochs2,
                  class_weight=cw, callbacks=callbacks(name, "phase2"))

    model.save(MODELS_DIR / f"{name}.keras")    # modèle final utilisé partout ensuite
    print("Modèle sauvegardé :", MODELS_DIR / f"{name}.keras")
    tracer_courbes(name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Entraîne un modèle de détection de pneumonie.")
    p.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "resnet50", "vgg19"])
    p.add_argument("--epochs1", type=int, default=15, help="epochs de la phase 1 (base gelée)")
    p.add_argument("--epochs2", type=int, default=10, help="epochs de la phase 2 (fine-tuning), 0 = aucune")
    a = p.parse_args()
    entrainer(a.model, a.epochs1, a.epochs2)
