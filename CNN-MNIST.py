import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load MNIST dataset
# To use Fashion MNIST instead, replace the line below with:
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Data preprocessing
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Optional: use class names for Fashion MNIST
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display some samples
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    # Uncomment below to show class names if using Fashion MNIST
    # plt.title(class_names[y_train[i]])
    plt.title(f"Label: {y_train[i]}")
plt.suptitle("Sample Training Images")
plt.tight_layout()
plt.show()

# Define CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train original model
model_original = build_model()
history_original = model_original.fit(
    x_train, y_train, epochs=3, batch_size=64, validation_split=0.1
)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False  # Set to False for digit datasets like MNIST
)
datagen.fit(x_train)

# Manually create validation split for augmented training
val_split = 0.1
val_size = int(len(x_train) * val_split)

x_val = x_train[:val_size]
y_val = y_train[:val_size]
x_train_aug = x_train[val_size:]
y_train_aug = y_train[val_size:]

# Train model with augmented data
model_augmented = build_model()
history_augmented = model_augmented.fit(
    datagen.flow(x_train_aug, y_train_aug, batch_size=64),
    validation_data=(x_val, y_val),
    epochs=3,
    steps_per_epoch=len(x_train_aug) // 64
)


# Predictions on test set (original model)
y_pred = model_original.predict(x_test).argmax(axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Original Model")  # Update for Fashion MNIST if needed
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Plot training & validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history_original.history['accuracy'], label='Original - Training')
plt.plot(history_original.history['val_accuracy'], label='Original - Validation')
plt.plot(history_augmented.history['accuracy'], label='Augmented - Training')
plt.plot(history_augmented.history['val_accuracy'], label='Augmented - Validation')
plt.title('Training & Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
