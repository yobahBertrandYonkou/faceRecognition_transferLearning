#!/usr/bin/env python
# coding: utf-8

# In[1]:


#required libraries
import cv2 as cv
from keras.models import load_model
from keras.applications.mobilenet import decode_predictions
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
import numpy as np


# In[2]:


#loads haar cascade frontal face detector
haar = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[3]:


#loads model created form transfer learning
model = load_model('myRecog.h5')


# In[4]:


#takes a photo and classifies the photo
def predict(photo):
    photo = cv.resize(photo, (224,224))
    photo = image.img_to_array(photo)
    photo = preprocess_input(photo)
    photo = np.expand_dims(photo, axis = 0)
    pred = model.predict(photo)
    return pred


# In[5]:


#prediction holds the percetage of certainty for each class in the list people
predictions = {}
people = ['Mishan', 'Siam', 'Yobah']


# In[6]:


#gets the key of the element with the highest certainty value
def getKey(pred):
    j = 0
    m = max(predictions.values())
    for i in pred.values():
        if i == m:
            return j, m
        else:
            j += 1


# In[11]:


pred = []
camera = cv.VideoCapture(0) #creates a video object for my cam
while True:
    _, photo = camera.read() #clicks a photo
    faces = haar.detectMultiScale(photo)  #detects faces from the above photo
    
    if len(faces) == 1: 
        #sends photo to prediction function if only one face is found
        pred = predict(photo) 
        j = 0
        #inserts categories as keys and percentage of certainty as values
        for layer in pred:
            for percentage in layer:
                predictions[people[j]] = round((percentage * 100), 2) 
                j+=1
        #gets the index and value with the highest percentage of certainty 
        index, value = getKey(predictions)
        #outputs the name of the person with the hieghest certainty if value is greater than 90
        if value > 90: 

            cv.putText(photo, f'{people[index]} {value} %',
                   (25,25), cv.FONT_HERSHEY_SIMPLEX, 1,
                   [0,255,0], 2, cv.LINE_AA)
    #draws a green rectangle around detected faces
    for (top, right, bottom, left) in faces:
        x1 = top
        y1 = right
        x2 = x1+bottom
        y2 = y1+left
        
        photo = cv.rectangle(photo, (x1, y1), (x2, y2), [0,225,0], 2)
    #displayes photos
    cv.imshow('video', photo)
    #keeps photo windows open until the enter key is pressed
    if cv.waitKey(1) == 13:
        #destroys image windows if enter is pressed
        cv.destroyAllWindows()
        break;
#releases camera from object
camera.release()

