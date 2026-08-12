import tensorflow as tf
from tensorflow.keras import layers, models


NUM_CLASSES = 10


def build_cnn():

    model = models.Sequential([
        
        layers.Input(shape=(32, 32, 3)),

        # Convolution Block 1
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolution Block 2
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolution Block 3
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classification Head
        layers.Flatten(),

        layers.Dense(
            256,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    return model


if __name__ == "__main__":

    model = build_cnn()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("=" * 60)
    print("CNN MODEL ARCHITECTURE")
    print("=" * 60)

    model.summary()

    print("\nCNN Model Created Successfully.")