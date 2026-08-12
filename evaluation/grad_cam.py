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
true_label = int(labels[0].numpy())


# ---------------------------------------------------------
# Find EfficientNet base model
# ---------------------------------------------------------

base_index = None
base_model = None

for i, layer in enumerate(model.layers):

    if isinstance(layer, tf.keras.Model):

        if "efficientnet" in layer.name.lower():

            base_index = i
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
# Find last convolutional layer
# ---------------------------------------------------------

last_conv_layer = None

for layer in reversed(base_model.layers):

    if isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D
        )
    ):

        last_conv_layer = layer
        break


if last_conv_layer is None:
    raise ValueError(
        "Convolutional layer not found."
    )


print(
    f"Grad-CAM layer: {last_conv_layer.name}"
)


# ---------------------------------------------------------
# Create activation model
# ---------------------------------------------------------

activation_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)


# ---------------------------------------------------------
# Run preprocessing layers
# ---------------------------------------------------------

x = image

for layer in model.layers[:base_index]:

    if isinstance(layer, tf.keras.layers.InputLayer):
        continue

    x = layer(x)


# ---------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------

print("\nGenerating Grad-CAM...")

with tf.GradientTape() as tape:

    conv_outputs, base_outputs = activation_model(x)

    # Apply layers after EfficientNet
    predictions_input = base_outputs

    for layer in model.layers[base_index + 1:]:

        predictions_input = layer(
            predictions_input,
            training=False
        )

    predictions = predictions_input

    predicted_class = tf.argmax(
        predictions[0]
    )

    class_score = predictions[
        0,
        predicted_class
    ]


# ---------------------------------------------------------
# Calculate gradients
# ---------------------------------------------------------

gradients = tape.gradient(
    class_score,
    conv_outputs
)


if gradients is None:

    raise ValueError(
        "Gradients could not be calculated."
    )


pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(1, 2)
)


conv_outputs = conv_outputs[0]

pooled_gradients = pooled_gradients[0]


heatmap = tf.reduce_sum(
    conv_outputs * pooled_gradients,
    axis=-1
)


heatmap = tf.maximum(
    heatmap,
    0
)


heatmap = heatmap / (
    tf.reduce_max(heatmap) + 1e-8
)


heatmap = heatmap.numpy()


predicted_class = int(
    predicted_class.numpy()
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


# Resize heatmap to image size
heatmap_resized = tf.image.resize(
    heatmap[..., np.newaxis],
    (32, 32)
).numpy().squeeze()


# ---------------------------------------------------------
# Create visualization
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 4)
)


plt.subplot(1, 2, 1)

plt.imshow(
    original_image
)

plt.title(
    f"Original\n"
    f"True: {CLASS_NAMES[true_label]}"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    original_image
)

plt.imshow(
    heatmap_resized,
    cmap="jet",
    alpha=0.5
)

plt.title(
    f"Grad-CAM\n"
    f"Predicted: {CLASS_NAMES[predicted_class]}"
)

plt.axis("off")


plt.tight_layout()


plt.savefig(
    "evaluation/figures/grad_cam.png",
    dpi=300
)


plt.close()


print("=" * 60)
print("GRAD-CAM COMPLETED")
print("=" * 60)

print(
    "Grad-CAM saved successfully."
)

print(
    "evaluation/figures/grad_cam.png"
)

print(
    "\nGrad-CAM Visualization "
    "Completed Successfully."
)