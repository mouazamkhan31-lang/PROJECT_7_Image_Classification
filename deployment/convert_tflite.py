import os
import tensorflow as tf


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "models/saved/efficientnet_finetuned.keras"

OUTPUT_DIR = "deployment"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "efficientnetb0_model.tflite"
)


# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 60)
print("TENSORFLOW LITE MODEL CONVERSION")
print("=" * 60)

print("\nLoading trained EfficientNetB0 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    safe_mode=False
)

print("Model loaded successfully!")


# ==========================================================
# CONVERT MODEL
# ==========================================================

print("\nConverting model to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

tflite_model = converter.convert()


# ==========================================================
# SAVE MODEL
# ==========================================================

with open(
    OUTPUT_PATH,
    "wb"
) as f:

    f.write(
        tflite_model
    )


# ==========================================================
# DISPLAY RESULT
# ==========================================================

file_size_mb = (
    os.path.getsize(OUTPUT_PATH)
    / (1024 * 1024)
)


print("\n" + "=" * 60)
print("TFLITE CONVERSION COMPLETED")
print("=" * 60)

print(
    f"\nTFLite model saved to:"
)

print(
    OUTPUT_PATH
)

print(
    f"\nModel size: {file_size_mb:.2f} MB"
)

print(
    "\nTensorFlow Lite deployment model created successfully!"
)