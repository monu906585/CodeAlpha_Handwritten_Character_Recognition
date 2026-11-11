"""Train a CNN on MNIST handwritten digits and save the trained model."""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load the MNIST dataset: 60,000 training images and 10,000 test images.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0-1 and add the channel dimension.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# CNN learns visual features such as edges, curves, and digit shapes.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Stop early if validation performance stops improving.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=2, restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=128,
    callbacks=[early_stop],
)

# Evaluate the trained model on test images.
probabilities = model.predict(x_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest accuracy: {accuracy:.4f}")
print(classification_report(y_test, predictions))

# Save the trained model for use in the Streamlit application.
model.save("models/handwritten_digit_model.keras")

# Save the training/validation accuracy graph.
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/training_accuracy.png")
print("Model saved to models/handwritten_digit_model.keras")
