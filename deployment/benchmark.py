import tensorflow as tf
import numpy as np
import time


MODEL_PATH = "deployment/efficientnetb0_model.tflite"


print("=" * 60)
print("TFLITE INFERENCE SPEED BENCHMARK")
print("=" * 60)


# Load model
interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("\nModel loaded successfully!")


# Create sample input
input_shape = input_details[0]["shape"]

sample_image = np.random.rand(
    *input_shape
).astype(np.float32)


# Warm-up inference
for _ in range(5):

    interpreter.set_tensor(
        input_details[0]["index"],
        sample_image
    )

    interpreter.invoke()


# Benchmark
runs = 20

start_time = time.perf_counter()


for _ in range(runs):

    interpreter.set_tensor(
        input_details[0]["index"],
        sample_image
    )

    interpreter.invoke()


end_time = time.perf_counter()


total_time = end_time - start_time

average_time = (
    total_time / runs
)

average_ms = (
    average_time * 1000
)


print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print(
    f"\nTotal runs: {runs}"
)

print(
    f"Total time: {total_time:.4f} seconds"
)

print(
    f"Average inference time: {average_ms:.2f} ms"
)


if average_ms < 100:

    print(
        "\nSTATUS: PASS"
    )

    print(
        "Inference time is below 100ms per image."
    )

else:

    print(
        "\nSTATUS: ABOVE TARGET"
    )

    print(
        "Inference time is above 100ms per image."
    )


print(
    "\nTFLite inference benchmark completed successfully!"
)