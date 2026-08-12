from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io


# ==========================================================
# CONFIGURATION
# ==========================================================

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


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading EfficientNetB0 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    safe_mode=False
)

print("Model loaded successfully!")


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((32, 32))

    image = np.array(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ==========================================================
# HOME / HEALTH CHECK
# ==========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "online",
        "message": "Advanced Image Classification API",
        "model": "EfficientNetB0",
        "classes": CLASS_NAMES
    })


# ==========================================================
# SINGLE IMAGE PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "error": "No image file provided."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "success": False,
            "error": "No image selected."
        }), 400

    try:

        image = Image.open(
            io.BytesIO(file.read())
        )

        processed_image = preprocess_image(
            image
        )

        predictions = model.predict(
            processed_image,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index]
        )

        top_indices = np.argsort(
            predictions
        )[::-1][:5]

        top_predictions = []

        for index in top_indices:

            top_predictions.append({
                "class": CLASS_NAMES[index],
                "confidence": round(
                    float(predictions[index]) * 100,
                    2
                )
            })

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "confidence": round(
                confidence * 100,
                2
            ),
            "top_5_predictions": top_predictions
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==========================================================
# BATCH IMAGE PREDICTION
# ==========================================================

@app.route("/predict_batch", methods=["POST"])
def predict_batch():

    files = request.files.getlist("images")

    if not files:

        return jsonify({
            "success": False,
            "error": "No images provided."
        }), 400

    results = []

    for file in files:

        if file.filename == "":
            continue

        try:

            image = Image.open(
                io.BytesIO(file.read())
            )

            processed_image = preprocess_image(
                image
            )

            predictions = model.predict(
                processed_image,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(predictions)
            )

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            confidence = float(
                predictions[predicted_index]
            )

            top_indices = np.argsort(
                predictions
            )[::-1][:5]

            top_predictions = []

            for index in top_indices:

                top_predictions.append({
                    "class": CLASS_NAMES[index],
                    "confidence": round(
                        float(predictions[index]) * 100,
                        2
                    )
                })

            results.append({
                "filename": file.filename,
                "prediction": predicted_class,
                "confidence": round(
                    confidence * 100,
                    2
                ),
                "top_5_predictions": top_predictions
            })

        except Exception as e:

            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return jsonify({
        "success": True,
        "total_images": len(results),
        "results": results
    })


# ==========================================================
# START SERVER
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("IMAGE CLASSIFICATION API")
    print("=" * 60)

    print("Server starting...")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )