import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data.data_loader import create_datasets


MODEL_PATH = "models/saved/efficientnet_finetuned.keras"

os.makedirs(
    "evaluation/figures",
    exist_ok=True
)


print("=" * 60)
print("LOADING FINE-TUNED MODEL")
print("=" * 60)


model = tf.keras.models.load_model(
    MODEL_PATH,
    safe_mode=False
)


_, _, test_ds = create_datasets()

images, labels = next(iter(test_ds))

image = images[0:1]

# EfficientNetB0 requires 224x224 images
image = tf.image.resize(
    image,
    (224, 224)
)


# ---------------------------------------------------------
# Find EfficientNet model
# ---------------------------------------------------------

base_model = None

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        if "efficientnet" in layer.name.lower():

            base_model = layer
            break


if base_model is None:

    raise ValueError(
        "EfficientNet base model not found."
    )


print(
    f"EfficientNet model found: {base_model.name}"
)


# ---------------------------------------------------------
# Find useful convolutional layer
# ---------------------------------------------------------

selected_layer = None

for layer in reversed(base_model.layers):

    if isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D
        )
    ):

        try:

            shape = layer.output.shape

            if (
                len(shape) == 4
                and shape[1] is not None
                and shape[2] is not None
                and shape[1] > 1
                and shape[2] > 1
            ):

                selected_layer = layer
                break

        except Exception:

            continue


if selected_layer is None:

    raise ValueError(
        "Suitable feature map layer not found."
    )


print(
    f"Feature Map Layer: "
    f"{selected_layer.name}"
)


# ---------------------------------------------------------
# Create feature extractor
# ---------------------------------------------------------

feature_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=selected_layer.output
)


print(
    "\nGenerating feature maps..."
)


features = feature_model(
    image,
    training=False
)


features = features[0].numpy()


print(
    f"Feature map shape: "
    f"{features.shape}"
)


# ---------------------------------------------------------
# Normalize feature maps
# ---------------------------------------------------------

num_maps = min(
    16,
    features.shape[-1]
)


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

plt.figure(
    figsize=(12, 12)
)


for i in range(num_maps):

    feature_map = features[:, :, i]

    feature_map = (
        feature_map
        - feature_map.min()
    )

    max_value = feature_map.max()

    if max_value > 0:

        feature_map = (
            feature_map / max_value
        )


    plt.subplot(
        4,
        4,
        i + 1
    )

    plt.imshow(
        feature_map,
        cmap="viridis"
    )

    plt.title(
        f"Feature {i + 1}"
    )

    plt.axis("off")


plt.suptitle(
    f"EfficientNetB0 Feature Maps\n"
    f"Layer: {selected_layer.name}"
)


plt.tight_layout()


plt.savefig(
    "evaluation/figures/feature_maps.png",
    dpi=300
)


plt.close()


print("=" * 60)
print("FEATURE MAP VISUALIZATION COMPLETED")
print("=" * 60)


print(
    "Feature maps saved successfully."
)


print(
    "evaluation/figures/feature_maps.png"
)


print(
    "\nFeature Map Visualization "
    "Completed Successfully."
)