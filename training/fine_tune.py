import os
import tensorflow as tf

from models.transfer_model import build_transfer_model
from data.data_loader import create_datasets


EPOCHS = 2

os.makedirs("models/saved", exist_ok=True)

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

train_ds, val_ds, test_ds = create_datasets()

print("\nBuilding EfficientNetB0...")

model = build_transfer_model()

# Load the best transfer-learning model
model.load_weights(
    "models/saved/efficientnet_best.keras"
)

# Find EfficientNet base model
base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ) and "efficientnet" in layer.name.lower():

        base_model = layer
        break


if base_model is None:

    raise ValueError(
        "EfficientNet base model not found."
    )


print(
    "\nEfficientNet base model found:"
)

print(
    base_model.name
)


# Unfreeze the base model
base_model.trainable = True


# Freeze most layers
# Only fine-tune the last 20 layers

for layer in base_model.layers[:-20]:

    layer.trainable = False


print(
    "\nFine-tuning last 20 layers..."
)


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=1,

        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(

        "models/saved/efficientnet_finetuned.keras",

        monitor="val_accuracy",

        save_best_only=True
    )
]


print("=" * 60)
print("FINE-TUNING STARTED")
print("=" * 60)


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks
)


print("=" * 60)
print("EVALUATING FINE-TUNED MODEL")
print("=" * 60)


test_loss, test_accuracy = model.evaluate(

    test_ds,

    verbose=1
)


print(
    f"\nFine-Tuned Test Loss: "
    f"{test_loss:.4f}"
)


print(
    f"Fine-Tuned Test Accuracy: "
    f"{test_accuracy:.4f}"
)


print(
    f"Fine-Tuned Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


print(
    "\nEfficientNetB0 Fine-Tuning "
    "Completed Successfully."
)