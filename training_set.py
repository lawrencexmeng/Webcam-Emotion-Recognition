import os
import random
import numpy as np
import pickle
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import load_img
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import warnings
from tqdm import tqdm
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
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
trainFrame = trainFrame.sample(frac = 1).reset_index(drop = True)
print(trainFrame.head())

testFrame = pd.DataFrame()

testFrame['image'], testFrame['label'] = loadDataset(TEST_DIR)
print(testFrame.head())

def featureExtract(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode = 'grayscale')
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features


trainFeatures = featureExtract(trainFrame['image']) 
testFeatures = featureExtract(testFrame['image'])

x_train = trainFeatures/255.0 #Normalize from 0 to 1
x_test = testFeatures/255.0

le = LabelEncoder() #Creation of label encoder object
le.fit(trainFrame['label']) #Identify unique labels with fit
y_train = le.transform(trainFrame['label']) #Turn unique labels into respective numerical forms with transform
y_test = le.transform(testFrame['label'])


input_shape = (48, 48, 1) #Shape of each image is 48x48 with channel=1
output_class = 7 #7 classes of emotion

y_train = to_categorical(y_train, num_classes=7) #One-hot encoding
y_test = to_categorical(y_test, num_classes=7) 

model_save_name = 'emotion_detection_model.keras'
history_save_name = 'training_history.pkl'
current_directory = os.getcwd()
model_save_path = os.path.join(current_directory, model_save_name)
history_save_path = os.path.join(current_directory, history_save_name)
history = None

if os.path.exists(model_save_path):
    print(f"Model file {model_save_name} already exists. Loading the model.")
    model = load_model(model_save_path)
else:
    print(f"Model file {model_save_name} does not exist. Training the model.")
#Convolutional Neural Net

    model = Sequential()
    #Convolutional Layers
    model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu' ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.3))

    model.add(Dense(output_class, activation = 'softmax'))

    model.compile(optimizer  = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(x=x_train, y=y_train, batch_size = 128, epochs = 100, validation_data = (x_test, y_test))
    model.save(model_save_path)

    with open(history_save_path, 'wb') as file:
        pickle.dump(history.history, file)      


model = load_model('emotion_detection_model.keras')
image_index = random.randint(0, len(testFrame))
print("Original Output: ", testFrame['label'][image_index])
pred = model.predict(x_test[image_index].reshape(1, 48, 48, 1))
prediction_label = le.inverse_transform([pred.argmax()])[0]
print("Predicted Output: ", prediction_label)
plt.imshow(x_test[image_index].reshape(48, 48), cmap ='gray')
plt.show()