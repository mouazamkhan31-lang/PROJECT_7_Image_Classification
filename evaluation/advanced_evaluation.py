import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

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

NUM_CLASSES = 10


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


print("\nGenerating predictions...")


predictions = model.predict(
    test_ds,
    verbose=1
)


y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])


y_pred = np.argmax(
    predictions,
    axis=1
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("=" * 60)
print("GENERATING CONFUSION MATRIX")
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


plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.title(
    "EfficientNetB0 Fine-Tuned Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    "evaluation/figures/confusion_matrix.png",
    dpi=300
)

plt.close()


print(
    "Confusion Matrix saved successfully."
)


# ==========================================================
# MULTI-CLASS ROC CURVE
# ==========================================================

print("=" * 60)
print("GENERATING MULTI-CLASS ROC CURVES")
print("=" * 60)


y_true_binary = label_binarize(
    y_true,
    classes=np.arange(NUM_CLASSES)
)


plt.figure(
    figsize=(10, 8)
)


for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(
        y_true_binary[:, i],
        predictions[:, i]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Multi-Class ROC Curves - EfficientNetB0"
)

plt.legend(
    loc="lower right",
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "evaluation/figures/roc_curves.png",
    dpi=300
)

plt.close()


print(
    "ROC Curves saved successfully."
)


print("=" * 60)
print("ADVANCED EVALUATION COMPLETED")
print("=" * 60)

print(
    "\nFiles created:"
)

print(
    "evaluation/figures/confusion_matrix.png"
)

print(
    "evaluation/figures/roc_curves.png"
)

print(
    "\nConfusion Matrix + ROC Evaluation "
    "Completed Successfully."
)