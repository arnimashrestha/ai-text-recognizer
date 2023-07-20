import os
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import  matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


#setting up the files, and checking that everything is good

#opening up the data by using pandas
nepalidata = pd.read_csv(os.environ["datadownnloads"])
#print(nepalidata.head())


#finding the length of the nepali data
#print("len", len(nepalidata))


from sklearn import preprocessing
#label encoding because these are non integer values so comp will not understand
lbl_encode= preprocessing.LabelEncoder()
lbl_encode.fit_transform(nepalidata['character'])


labelencoder = preprocessing.LabelEncoder()
nepalidata['character'] = labelencoder.fit_transform(nepalidata['character'])


#label encoding(because the categories on the data set are non-numericals, and the program works better with numerical values)
y= nepalidata.character
x=nepalidata.drop('character', axis=1)
#print(y.head())#says what the letters are(is not binary) 0 is ka, 1 is kha


#splitting the data
trainingData, testingData, trainingLabels, testingLabels = train_test_split(x/255, y, test_size=0.2)


#creating the layers
model = keras.Sequential([
    keras.layers.Dense(1024, activation='sigmoid'),
    keras.layers.Dense(2048, activation='sigmoid'),
    keras.layers.Dense(46, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#training the model
model.fit(trainingData, trainingLabels, validation_data=(testingData, testingLabels), epochs=5)
test_loss, test_acc = model.evaluate(testingData, testingLabels, verbose=2)
print(test_loss, test_acc)


#saving the model
model.save(os.environ["datadownloads"])