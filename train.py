import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from architecture import *

# === 1. Wczytaj dane MNIST ===
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:1000]
y_train = y_train[:1000]

# Rozszerz do 3 kanałów RGB
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

X_train = np.repeat(X_train, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# Resize do 160x160 (rozmiar FaceNet)
X_train = tf.image.resize(X_train, [160, 160]).numpy()
X_test = tf.image.resize(X_test, [160, 160]).numpy()

# Normalizacja (zgodnie z ImageNet / FaceNet)
X_train = tf.keras.applications.imagenet_utils.preprocess_input(X_train)
X_test = tf.keras.applications.imagenet_utils.preprocess_input(X_test)

# One-hot encoding etykiet
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# === 2. Załaduj model FaceNet ===
print("Ładowanie modelu FaceNet...")
facenet = InceptionResNetV2()
path = "facenet_keras_weights.h5"
facenet.load_weights(path)
facenet.trainable = False  # Zamrożenie wag

# === 3. Zbuduj nowy model z warstwą klasyfikacji ===
inputs = Input(shape=(160, 160, 3))
x = facenet(inputs)
outputs = Dense(10, activation="softmax")(x)

model = Model(inputs, outputs)

# === 4. Kompiluj i trenuj ===
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Trenowanie modelu...")
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.15)

# === 5. Ocena ===
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {acc:.4f}")
