import tensorflow as tf
from tensorflow.keras import layers


def create_augmentation_pipeline():

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal"
            ),

            layers.RandomRotation(
                0.1
            ),

            layers.RandomZoom(
                0.1
            ),

            layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1
            ),

            layers.RandomContrast(
                0.1
            )
        ],
        name="data_augmentation"
    )

    return augmentation


if __name__ == "__main__":

    augmentation = create_augmentation_pipeline()

    print("=" * 60)
    print("DATA AUGMENTATION PIPELINE")
    print("=" * 60)

    augmentation.summary()

    print(
        "\nData Augmentation Pipeline "
        "Created Successfully."
    )