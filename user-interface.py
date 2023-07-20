#importing the libraries needed 
import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from PIL import Image


workingImage = np.zeros((500,500,3), dtype='float64') 
nepalicharName= ["ka", "kha", "ga", "gha", "nga", "cha", " chaa", "ja", "jha", "yuh", "ta", "tha", "da", "dha", "uhda", "ta", "tha", "da", "dha", "na", "pa","pha", "ba", "bha", "ma", "ya", "ra", "la", "wa", "sa", "sa", "sha", "ha", "chay", "tra", "gya", 1, 2,3, 4, 5, 6, 7, 8, 9]
run= False #run will be false, global function
def trackdraw(event, x, y, flag, param ):
    global run
    if event==cv2.EVENT_LBUTTONDOWN: #this indicates that the right button is pressed down
        run=True
        cv2.circle(workingImage, (x,y), 20, (255,255,255), -10 )
    if event==cv2.EVENT_LBUTTONUP: #if the button is released we dont want to do anything
        run=False
    if event==cv2.EVENT_MOUSEMOVE: #this is when we move the mouse
        if run==True: #run will be true, and it will show the image
            cv2.circle(workingImage, (x,y),20,(255,255,255),-10) #changing the colour to WHITE which is 255,255,255 and the - basically fills
            #the - basically fills in the image
#this basically allows the window to show on the image
cv2.namedWindow('window') #what the window will be called 
cv2.setMouseCallback('window', trackdraw)




model = keras.models.load_model('C:/Users/binju/TrainData')
while True:
    img = cv2.imshow('window', workingImage)
    k = cv2.waitKey(1) 
    if k==ord('c'):
        workingImage = np.zeros((500,500,3), dtype='float64') 
   
    if k==ord('s'):
        workingImage = cv2.resize(workingImage, (32, 32))
        cv2.imwrite("nepali_character.jpg", workingImage)
        cv2.imshow("resized", workingImage)
        grayscaleInput = np.asarray(cv2.imread("nepali_character.jpg", 0)).astype('float64').flatten() / 255
        grayscaleInput = np.expand_dims(grayscaleInput, axis=0)
        print(type(grayscaleInput))
        print("shape =", grayscaleInput.shape)
        predictions= model.predict(grayscaleInput)
        print(predictions)
        numchar= np.argmax(predictions)
        print("numchar=", numchar)
        print("result:", nepalicharName[numchar], "certainty:", predictions[0][numchar] * 100, "%")
        break
    if k==ord('q'): #if k is 27 then we break the window
        cv2.destroyAllWindows()
        break