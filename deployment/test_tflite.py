import tensorflow as tf
import numpy as np
from PIL import Image


MODEL_PATH = "deployment/efficientnetb0_model.tflite"

IMAGE_PATH = r"C:\Users\hp\OneDrive\Pictures\desktop-slider-sale-pk-5_66dd8983-d298-4f30-acf8-15ceece627cc.webp"

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


print("=" * 60)
print("TENSORFLOW LITE MODEL TEST")
print("=" * 60)


# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("\nTFLite model loaded successfully!")

print(
    "Input shape:",
    input_details[0]["shape"]
)


# Load image
image = Image.open(
    IMAGE_PATH
).convert("RGB")


image = image.resize(
    (32, 32)
)


image = np.array(
    image,
    dtype=np.float32
) / 255.0


image = np.expand_dims(
    image,
    axis=0
)


# Run inference
interpreter.set_tensor(
    input_details[0]["index"],
    image
)


interpreter.invoke()


output = interpreter.get_tensor(
    output_details[0]["index"]
)[0]


# Prediction
predicted_index = int(
    np.argmax(output)
)


predicted_class = CLASS_NAMES[
    predicted_index
]


confidence = float(
    output[predicted_index]
)


print("\n" + "=" * 60)
print("TFLITE PREDICTION")
print("=" * 60)

print(
    "\nPredicted Class:",
    predicted_class.upper()
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)

print(
    "\nTFLite inference completed successfully!"
)