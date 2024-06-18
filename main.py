import cv2 
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from sklearn.preprocessing import LabelEncoder
  
# define a video capture object 
vid = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = load_model('emotion_detection_model.keras')
label_encoder = LabelEncoder()
label_encoder.fit(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])


def faceBox(frame):
    grayedImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grayedImage, 1.3, 10, minSize=(70, 70)) 
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 2)
    return faces

def preprocess(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

while(True): 
    ret, vidFrame = vid.read() #capturing frame from cam 0 
    if not ret:
        break
    faces = faceBox(vidFrame)
    
    for(x, y, w, h) in faces:
        face = vidFrame[y: y+h, x: x+w]
        processed_face = preprocess(face)
        pred = model.predict(processed_face)
        emotion = label_encoder.inverse_transform([np.argmax(pred)])
        emotion = str(emotion)
        cv2.putText(vidFrame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame 
    cv2.imshow('frame', vidFrame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #check every 1 MS if 'q' is entered, if so, exit while loop
        break

# After the loop release the cap object 
vid.release() 

# Destroy all the windows 
cv2.destroyAllWindows()