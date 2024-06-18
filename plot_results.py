import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import load_img
import numpy as np
from PIL import Image

# Load training and test frames
TRAIN_DIR = r'C:\Users\lawre\Desktop\FERP\train'
TEST_DIR = r'C:\Users\lawre\Desktop\FERP\test'

def loadDataset(directory):
    imgPaths = []
    labels = []
    for label in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, label)):
            imagePath = os.path.join(directory, label, filename)
            imgPaths.append(imagePath)
            labels.append(label)
        print(label, "Completed")
    return imgPaths, labels

trainFrame = pd.DataFrame()
trainFrame['image'], trainFrame['label'] = loadDataset(TRAIN_DIR)
trainFrame = trainFrame.sample(frac=1).reset_index(drop=True)
print(trainFrame.head())

testFrame = pd.DataFrame()
testFrame['image'], testFrame['label'] = loadDataset(TEST_DIR)
print(testFrame.head())

history_save_name = 'training_history.pkl'
current_directory = os.getcwd()
history_save_path = os.path.join(current_directory, history_save_name)

if os.path.exists(history_save_path):
    with open(history_save_path, 'rb') as file:
        history = pickle.load(file)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Accuracy Graph')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Loss Graph')
    plt.legend()
    plt.show()
else:
    print(f"History file {history_save_name} not found.")

# Distribution of emotions in training set
plt.figure()
sns.countplot(x=trainFrame['label'])
plt.show()

# First training set image
img = Image.open(trainFrame['image'][0])
plt.figure()
plt.imshow(img, cmap='gray')
plt.show()

# Random 25 from training set display
plt.figure(figsize=(8, 8))
files = trainFrame.iloc[0:25]

for index, file, label in files.itertuples():
    plt.subplot(5, 5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')
plt.tight_layout()
plt.show()
