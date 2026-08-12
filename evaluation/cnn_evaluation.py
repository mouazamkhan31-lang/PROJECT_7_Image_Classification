import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from data.data_loader import create_datasets


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


MODEL_PATH = "models/saved/cnn_best.keras"


print("=" * 60)
print("LOADING TRAINED CNN MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

_, _, test_ds = create_datasets()


print("\nGenerating predictions...")

predictions = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])


print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )
)


print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("CNN Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "evaluation/cnn_confusion_matrix.png"
)

plt.show()


print(
    "\nCNN Evaluation Completed Successfully."
)

print(
    "Confusion matrix saved to:"
)

print(
    "evaluation/cnn_confusion_matrix.png"
)