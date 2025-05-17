from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

# Load the MNIST dataset
(X_train, y_train), (_, _) = mnist.load_data()

print(len(X_train[1]))
# Print 4 images in a row
plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()


# Załaduj dane
digits = load_digits()

# Wyświetl pierwsze 10 obrazów i ich etykiety
plt.figure(figsize=(10, 2))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(str(digits.target[i]))
    plt.axis("off")
plt.tight_layout()
plt.show()
