import tensorflow as tf
from tensorflow.keras import layers, models


NUM_CLASSES = 10


def build_transfer_model():

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # Pretrained layers freeze
    base_model.trainable = False

    inputs = layers.Input(
        shape=(32, 32, 3)
    )

    # Resize CIFAR-10 images
    x = layers.Resizing(
        224,
        224
    )(inputs)

    # EfficientNet expects image values in its normal input range
    x = layers.Lambda(
        lambda image: image * 255.0
    )(x)

    # Pretrained EfficientNet
    x = base_model(
        x,
        training=False
    )

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    return model


if __name__ == "__main__":

    model = build_transfer_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("=" * 60)
    print("TRANSFER LEARNING MODEL")
    print("=" * 60)

    model.summary()

    print(
        "\nEfficientNetB0 Transfer Learning "
        "Model Created Successfully."
    )