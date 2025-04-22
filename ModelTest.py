import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the model
model = tf.keras.models.load_model("cats_dogs_classifier.h5")

# Load an image (replace 'test_image.jpg' with your file)
#img_path = "C:\\Users\\Lenidar\\Downloads\\dog.jpg"
#img_path = "C:\\Users\\Lenidar\\Downloads\\dog2.jpg"
img_path = "C:\\Users\\Lenidar\\Downloads\\dog3.jpg"
#img_path = "C:\\Users\\Lenidar\\Downloads\\cat.jpg"
img = image.load_img(img_path, target_size=(128, 128))  # Resize to match model input

# Convert to array and normalize
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize

prediction = model.predict(img_array)

confidence = prediction if prediction > 0.5 else 1 - prediction

label = "Cat" if prediction[0][0] < 0.5 else "Dog"

print(prediction)

print(f"Prediction: {label} (Confidence: {float(confidence) * 100:.2f}%)")
