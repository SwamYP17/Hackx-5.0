import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing import image

# Load the trained model
with open('image_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to make predictions
def predict_forgery(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(img_array)

    # Determine if image is forged
    is_forged = prediction[0][0] > 0.5
    confidence = prediction[0][0] * 100

    return is_forged, confidence

# Folder containing new images for inference
folder_path = './new_images'  # Update this path to your custom dataset folder
output_data = []

# Loop through all images in the folder and make predictions
for img_file in os.listdir(folder_path):
    image_path = os.path.join(folder_path, img_file)
    if image_path.endswith(('.png', '.jpg', '.jpeg')):
        is_forged, confidence = predict_forgery(image_path)
        output_data.append([img_file, is_forged, confidence])

# Create a DataFrame for the results
output_df = pd.DataFrame(output_data, columns=['file_name', 'is_forged', 'confidence'])

# Save the output as a CSV file
output_df.to_csv('image_forgery_predictions.csv', index=False)

print("Predictions saved to 'image_forgery_predictions.csv'")