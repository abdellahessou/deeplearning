"""Étape 3 : construction des modèles par transfer learning (base ImageNet gelée + petite tête)."""
from tensorflow import keras
from tensorflow.keras import layers

from config import IMG_SIZE

# Les trois CNN pré-entraînés comparés dans le projet
BACKBONES = {
    "mobilenetv2": keras.applications.MobileNetV2,
    "resnet50": keras.applications.ResNet50,
    "vgg19": keras.applications.VGG19,
}

# Dernière couche convolutionnelle de chaque modèle (utilisée par Grad-CAM à l'étape 6)
LAST_CONV = {
    "mobilenetv2": "out_relu",
    "resnet50": "conv5_block3_out",
    "vgg19": "block5_conv4",
}


# Nombre de couches dégelées au fine-tuning (étape 4) : à peu près le dernier bloc
# convolutionnel de chaque modèle. VGG19 n'a que 22 couches : avec 30, on dégèlerait tout.
N_UNFREEZE = {
    "mobilenetv2": 30,   # blocs 14 à 16 + dernière convolution
    "resnet50": 30,      # étage conv5
    "vgg19": 5,          # bloc 5 (4 convolutions + pooling)
}


def build_model(name, dropout=0.3):
    """Renvoie (modèle complet, base pré-entraînée). La base est gelée."""
    inputs = keras.Input(shape=IMG_SIZE + (3,))             # image 224x224 à 3 canaux
    # Backbone pré-entraîné sans sa tête de classification ImageNet (1 000 classes).
    # input_tensor=inputs : ses couches font directement partie de notre modèle,
    # on pourra donc les retrouver par leur nom pour Grad-CAM.
    base = BACKBONES[name](include_top=False, weights="imagenet",
                           input_tensor=inputs, input_shape=IMG_SIZE + (3,))
    base.trainable = False                                  # on gèle toutes ses couches

    x = layers.GlobalAveragePooling2D()(base.output)        # 7x7xC → vecteur de C valeurs
    x = layers.Dropout(dropout)(x)                          # régularisation
    outputs = layers.Dense(1, activation="sigmoid")(x)      # probabilité de PNEUMONIA
    return keras.Model(inputs, outputs, name=name), base


def compile_model(model, lr):
    """Configure l'optimiseur, la loss et les métriques suivies pendant l'entraînement."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")])


def fine_tune(base, n_unfreeze=30):
    """Étape 4 : dégèle les n dernières couches de la base, BatchNorm toujours gelées."""
    base.trainable = True
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
