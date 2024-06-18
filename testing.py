import os
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from training_set import loadDataset

# Assuming you have the trainFrame DataFrame from previous code
TRAIN_DIR = r'C:\Users\lawre\Desktop\FERP\train'

# Load the labels from the training set to fit the LabelEncoder
trainFrame = pd.DataFrame()
trainFrame['image'], trainFrame['label'] = loadDataset(TRAIN_DIR)

# Load the model
model = load_model('emotion_detection_model.keras')

# Fit the LabelEncoder with the labels from the training set
le = LabelEncoder()
le.fit(trainFrame['label'])

def preprocess_image(image_path):
    # Load image in grayscale mode
    img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
    # Convert the image to array
    img = img_to_array(img)
    # Reshape the image to match the model input
    img = img.reshape(1, 48, 48, 1)
    # Normalize the pixel values
    img = img / 255.0
    return img

def predict_emotion(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    # Make prediction
    pred = model.predict(img)
    # Convert prediction to label
    prediction_label = le.inverse_transform([pred.argmax()])
    return prediction_label[0]

def main():
    # Test with a new image
    new_image_path = r'C:\Users\lawre\Desktop\FERP\smiling_woman.jpeg'  # Replace with your image path
    predicted_emotion = predict_emotion(new_image_path)
    print(f'Predicted Emotion: {predicted_emotion}')

    # Display the image
    img = load_img(new_image_path, color_mode='grayscale', target_size=(48, 48))
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted Emotion: {predicted_emotion}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
