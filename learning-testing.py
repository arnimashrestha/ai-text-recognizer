import os
import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn
model = keras.models.load_model(os.environ["traindata"])


#opening up the data by using pandas
nepalidata = pd.read_csv(os.environ["datadownloads"])
#print(nepalidata.head())


nepalicharName= ["ka", "kha", "ga", "gha", "nga", "cha", " chaa", "ja", "jha", "yuh", "ta", "tha", "da", "dha", "uhda", "ta", "tha", "da", "dha", "na", "pa","pha", "ba", "bha", "ma", "ya", "ra", "la", "wa", "sa", "sa", "sha", "ha", "chay", "tra", "gya", 1, 2,3, 4, 5, 6, 7, 8, 9]
#finding the length of the nepali data


from sklearn import preprocessing
#label encoding because these are non integer values so comp will not understand
lbl_encode= preprocessing.LabelEncoder()
lbl_encode.fit_transform(nepalidata['character'])


labelencoder = preprocessing.LabelEncoder()
nepalidata['character'] = labelencoder.fit_transform(nepalidata['character'])


#label encoding(because the categories on the data set are non-numericals, and the program works better with numerical values)
y= nepalidata.character
x=nepalidata.drop('character', axis=1)


from sklearn.model_selection import train_test_split
trainingData, testingData, trainingLabels, testingLabels = train_test_split(x/255, y, test_size=0.2)


testlist=[]
for rows in range(0, len(testingData)): #len will be around 18000
    indexofx_test= testingData.index[rows]
    testlist.append(indexofx_test)


#creating a function so that see what xtestrandom will be
def x_test_show(rownum):
    value= nepalidata.iloc[rownum] #iloc is a panda function
    #why do i need to do -1? for some reason its giving the value of+1
    rowxpixels= nepalidata.iloc[rownum]
    pixellist=[]
    lengthxtrain=len(rowxpixels)
    #prints it out as a list
    for pixels in rowxpixels[:lengthxtrain-1]:
        pixellist.append(pixels)
    print(pixellist)
    #cannot see that image through matplotlib, so have to reshape it into a matrix
    from numpy import array
    matrixpixel= np.array(pixellist)
    matrixp=(matrixpixel.reshape(32,32))
    print(matrixp)
    imgplot= plt.imshow(matrixp/255, cmap=plt.cm.binary)
    return (plt.show())


#testing the model with the trainingData pixels
y_predicted= model.predict(testingData)
print("list", y_predicted[900]) #num has to be within 0 and 18 400


#finding what the greatest weight of the number is
index= (np.argmax(y_predicted[900]))
print("inidex=", index)
number= (testlist[900])
print("number",  number)


#how to know that it's actually in the testlist, and it's printing something from the testlist
if number in testlist:
    print("yes")


print(number/2000)
print("index", nepalicharName[index])


num= testingData.index[900]
print(x_test_show(num))
(print(nepalicharName[index]))