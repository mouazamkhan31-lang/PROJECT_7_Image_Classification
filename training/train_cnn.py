import os
import tensorflow as tf

from models.cnn_model import build_cnn
from data.data_loader import create_datasets


EPOCHS = 10

os.makedirs("models/saved", exist_ok=True)

print("=" * 60)
print("LOADING CIFAR-10 DATASET")
print("=" * 60)

train_ds, val_ds, test_ds = create_datasets()

print("\nBuilding CNN model...")

model = build_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "models/saved/cnn_best.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]


print("=" * 60)
print("CNN TRAINING STARTED")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


print("=" * 60)
print("EVALUATING CNN MODEL")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(
    f"\nTest Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    "\nCNN Training Completed Successfully."
)