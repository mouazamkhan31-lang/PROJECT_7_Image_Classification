import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

from data.data_loader import create_datasets


MODEL_PATH = "models/saved/efficientnet_finetuned.keras"


print("=" * 60)
print("LOADING FINE-TUNED EFFICIENTNET")
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


# Top-1 Accuracy
top1_accuracy = accuracy_score(
    y_true,
    y_pred
)


# Top-5 Accuracy
top5_predictions = np.argsort(
    predictions,
    axis=1
)[:, -5:]

top5_correct = np.array([
    true_label in predicted_labels
    for true_label, predicted_labels
    in zip(y_true, top5_predictions)
])

top5_accuracy = np.mean(
    top5_correct
)


# Precision
precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# Recall
recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("=" * 60)
print("ADVANCED MODEL METRICS")
print("=" * 60)

print(
    f"Top-1 Accuracy: "
    f"{top1_accuracy * 100:.2f}%"
)

print(
    f"Top-5 Accuracy: "
    f"{top5_accuracy * 100:.2f}%"
)

print(
    f"Weighted Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Weighted Recall: "
    f"{recall * 100:.2f}%"
)


print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred
    )
)


print(
    "\nAdvanced Model Evaluation "
    "Completed Successfully."
)