import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "deployment/efficientnetb0_model.tflite"

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
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🖼️",
    layout="wide"
)


# ==========================================================
# LOAD TFLITE MODEL
# ==========================================================

@st.cache_resource
def load_model():

    interpreter = Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    return interpreter


interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "🖼️ Advanced Image Classification System"
)

st.subheader(
    "EfficientNetB0 Deep Learning Image Recognition"
)

st.markdown(
    """
    Upload an image and the trained EfficientNetB0
    model will classify it into one of the CIFAR-10 categories.
    """
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Settings")

top_k = st.sidebar.slider(
    "Number of Predictions",
    min_value=1,
    max_value=5,
    value=5
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Model: EfficientNetB0\n\n"
    "Fine-Tuned Accuracy: 90.15%"
)


# ==========================================================
# IMAGE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================================
# PREDICTION
# ==========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    col1, col2 = st.columns(2)


    # ------------------------------------------------------
    # DISPLAY IMAGE
    # ------------------------------------------------------

    with col1:

        st.markdown(
            "### 🖼️ Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    # ------------------------------------------------------
    # PREPROCESS IMAGE
    # ------------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.array(
        Image.fromarray(
            image_array.astype(np.uint8)
        ).resize(
            (32, 32)
        ),
        dtype=np.float32
    )

    image_array = (
        image_array / 255.0
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    with st.spinner(
        "🤖 Analyzing image..."
    ):

        interpreter.set_tensor(
            input_details[0]["index"],
            image_array
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(
            output_details[0]["index"]
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


    # ------------------------------------------------------
    # RESULT
    # ------------------------------------------------------

    with col2:

        st.markdown(
            "### 🎯 Prediction"
        )

        st.success(
            f"Predicted Class: **{predicted_class.upper()}**"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ======================================================
    # TOP-K PREDICTIONS
    # ======================================================

    st.markdown("---")

    st.markdown(
        "## 🏆 Top Predictions"
    )


    top_indices = np.argsort(
        predictions
    )[::-1][:top_k]


    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        class_name = CLASS_NAMES[
            index
        ]

        probability = (
            predictions[index] * 100
        )


        col1, col2, col3 = st.columns(
            [1, 3, 2]
        )


        with col1:

            st.write(
                f"### #{rank}"
            )


        with col2:

            st.write(
                f"**{class_name.upper()}**"
            )


        with col3:

            st.progress(
                float(
                    predictions[index]
                )
            )

            st.write(
                f"{probability:.2f}%"
            )


else:

    st.info(
        "👆 Upload an image above to start classification."
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Advanced Image Classification and Object Recognition "
    "System using CNN & EfficientNetB0"
)