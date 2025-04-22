import kagglehub
# tensorflow is not supported by python 3.13 go down to 3.12
# install scipy too
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download latest version
#path = kagglehub.dataset_download("marquis03/cats-and-dogs")

path = "C:\\Users\\Lenidar\\.cache\\kagglehub\\datasets\\marquis03\\cats-and-dogs\\versions\\3"

print("Path to dataset files:", path)

# Define dataset path (update this)
dataset_path = path

# Image settings
img_size = (128, 128)
batch_size = 32

# Data Preprocessing & Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

print(train_data.class_indices)

# Enhanced CNN Model Structure
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Reduces overfitting
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification (Cats vs Dogs)
])

# Adjust Learning Rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile Model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_data, validation_data=val_data, epochs=30)

# Save Model
model.save("cats_dogs_classifier_v2.h5")

print("Training complete! Model saved.")

loss, accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {accuracy:.2f}, Loss: {loss:.4f}")

print("Training complete! Model saved.")
