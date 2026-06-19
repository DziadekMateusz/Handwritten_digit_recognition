import os
import glob
import time
from pathlib import Path

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Mode
TRAIN_MODE = False   # True = train and save, False = load models

MODEL_PATHS = {
    "mlp": "mlp_model.keras",
    "cnn": "cnn_model.keras",
}


# Configuration
IMG_SIZE = (28, 28)
NUM_CLASSES = 10
BATCH_SIZE = 16
EPOCHS = 20

DATASET_DIR = "./dataset"



# Data loading
def load_dataset_structure(dataset_dir):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for digit in range(10):
        files = glob.glob(os.path.join(dataset_dir, str(digit), "*.png"))

        for filepath in files:
            try:
                img = Image.open(filepath).convert("L").resize(IMG_SIZE)
                arr = (np.array(img, dtype=np.float32) / 255.0).reshape(28, 28, 1)

                if "test" in Path(filepath).stem:
                    X_test.append(arr)
                    y_test.append(digit)
                else:
                    X_train.append(arr)
                    y_train.append(digit)

            except Exception as e:
                print(f"Skipped {filepath}: {e}")

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test)
    )

# Model
def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_mlp():
    return compile_model(models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(10, activation="softmax")
    ]))


def build_cnn():
    return compile_model(models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]))


BUILDERS = {"mlp": build_mlp, "cnn": build_cnn}

# Main
def main():
    X_train, y_train, X_test, y_test = load_dataset_structure(DATASET_DIR)

    if len(X_test) == 0:
        raise ValueError("No training data!")

    models_dict = {}

    if TRAIN_MODE:
        print("=== MODE: TRAINING ===")

        if len(X_train) == 0:
            raise ValueError("No training data!")

        y_train_cat = to_categorical(y_train, NUM_CLASSES)

        for name, build_fn in BUILDERS.items():
            print(f"\n=== TRAINING {name.upper()} ===")
            model = build_fn()
            model.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
            model.save(MODEL_PATHS[name])
            print(f"Saved {name.upper()} to: {MODEL_PATHS[name]}")
            models_dict[name] = model

    else:
        print("=== MODE: LOADING MODELS ===")

        if not all(os.path.exists(p) for p in MODEL_PATHS.values()):
            raise ValueError("Brak zapisanych modeli!")

        models_dict = {
            name: tf.keras.models.load_model(path)
            for name, path in MODEL_PATHS.items()
        }

        print("Models loaded properly.")

    model = models_dict["cnn"]

    # Test
    correct = 0
    times = []
    confidences = []

    print("\n=== PREDICTIONS ===")
    for i in range(len(X_test)):
        x = X_test[i:i + 1]

        start = time.perf_counter()
        probs = model.predict(x, verbose=0)[0]
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)

        pred_class = int(np.argmax(probs))
        true_class = int(y_test[i])

        correct += pred_class == true_class
        confidences.append(float(probs[pred_class]))

        print(f"{true_class}:{np.round(probs, 3).tolist()} Speed: {elapsed_ms:.0f} ms")

    avg_speed = np.mean(times)
    avg_confidence = np.mean(confidences) * 100.0

    accuracy_truncated = (correct * 10000 // len(X_test)) / 100

    print("\n=== SUMMARY ===")
    print(f"Accuracy: {accuracy_truncated:.2f}%")
    print(f"Avg confidence: {avg_confidence:.2f}%")
    print(f"Avg speed: {avg_speed:.0f} ms")


if __name__ == "__main__":
    main()