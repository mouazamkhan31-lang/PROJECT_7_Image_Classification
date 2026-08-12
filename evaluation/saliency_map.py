import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data.data_loader import create_datasets


MODEL_PATH = "models/saved/efficientnet_finetuned.keras"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

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

image = tf.cast(
    image,
    tf.float32
)


print("\nGenerating Saliency Map...")


# ---------------------------------------------------------
# Calculate gradients with respect to input image
# ---------------------------------------------------------

with tf.GradientTape() as tape:

    tape.watch(image)

    predictions = model(
        image,
        training=False
    )

    predicted_class = tf.argmax(
        predictions[0]
    )

    class_score = predictions[
        0,
        predicted_class
    ]


gradients = tape.gradient(
    class_score,
    image
)


if gradients is None:

    raise ValueError(
        "Saliency gradients could not be calculated."
    )


# ---------------------------------------------------------
# Convert gradients into saliency map
# ---------------------------------------------------------

saliency = tf.reduce_max(
    tf.abs(gradients),
    axis=-1
)


saliency = saliency[0].numpy()


# Normalize
saliency = (
    saliency - saliency.min()
)


max_value = saliency.max()

if max_value > 0:

    saliency = (
        saliency / max_value
    )


# ---------------------------------------------------------
# Get prediction
# ---------------------------------------------------------

predicted_class = int(
    predicted_class.numpy()
)

true_label = int(
    labels[0].numpy()
)


print(
    f"True Class: "
    f"{CLASS_NAMES[true_label]}"
)

print(
    f"Predicted Class: "
    f"{CLASS_NAMES[predicted_class]}"
)


# ---------------------------------------------------------
# Prepare original image
# ---------------------------------------------------------

original_image = image[0].numpy()

original_image = np.clip(
    original_image,
    0,
    1
)


# ---------------------------------------------------------
# Create visualization
# ---------------------------------------------------------

plt.figure(
    figsize=(12, 4)
)


plt.subplot(1, 3, 1)

plt.imshow(
    original_image
)

plt.title(
    f"Original\n"
    f"True: {CLASS_NAMES[true_label]}"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    saliency,
    cmap="hot"
)

plt.title(
    "Saliency Map"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    original_image
)

plt.imshow(
    saliency,
    cmap="jet",
    alpha=0.5
)

plt.title(
    f"Saliency Overlay\n"
    f"Predicted: {CLASS_NAMES[predicted_class]}"
)

plt.axis("off")


plt.tight_layout()


plt.savefig(
    "evaluation/figures/saliency_map.png",
    dpi=300
)


plt.close()


print("=" * 60)
print("SALIENCY MAP COMPLETED")
print("=" * 60)

print(
    "Saliency Map saved successfully."
)

print(
    "evaluation/figures/saliency_map.png"
)

print(
    "\nSaliency Map Visualization "
    "Completed Successfully."
)