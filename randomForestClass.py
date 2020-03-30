# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:31:46 2020

@author: Yohan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, balanced_accuracy_score,
                             fbeta_score, hamming_loss,recall_score,zero_one_loss,precision_score)
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import math



urlTrain = 'https://raw.githubusercontent.com/vardanagarwal/SIH/master/train.csv?token=AJP5RTWYQXWR4DWKYV7BHIK6RNHOK'
urlTest = 'https://raw.githubusercontent.com/vardanagarwal/SIH/master/test.csv?token=AJP5RTXYGDCSIANQSHX75YS6RNHQO'
dfTrain = pd.read_csv(urlTrain, error_bad_lines=False)
dfTest = pd.read_csv(urlTest, error_bad_lines=False)

X_train = dfTrain.drop(labels = ['CamId', 'Filename'], axis = 1).to_numpy()
y_train =  dfTrain['Div_labels'].to_numpy()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_test = dfTest.drop(labels = ['CamId', 'Filename'], axis = 1).to_numpy()
y_test =  dfTest['Div_labels'].to_numpy()

rf = RandomForestClassifier(n_jobs = -1)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)


filename = 'randomForestClass_model.pkl'
pickle.dump(rf, open(filename, 'wb'))


"""
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
"""


cm = confusion_matrix(y_test, y_pred)
cmScore = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/(cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+
                                             cm[0,1]+cm[0,2]+cm[0,3]+cm[1,0]+
                                             cm[1,2]+cm[1,3]+cm[2,0]+cm[2,1]+
                                             cm[2,3]+cm[3,0]+cm[3,1]+cm[3,2])
    #log = log_loss(y_pred, y_test)
ps = precision_score(y_test, y_pred, average = 'micro')
rs = recall_score(y_test, y_pred, average = 'micro')
f1 = f1_score(y_test, y_pred, average = 'micro')
bas = balanced_accuracy_score(y_test, y_pred)
fbs = fbeta_score(y_test, y_pred, beta=1, average = 'micro')
hl = hamming_loss(y_test, y_pred)
zol = zero_one_loss(y_test, y_pred)

data = [['CMScore', cmScore], ['Precision Score', ps], ['Recall Score', rs], 
        ['F1 Score', f1], ['Balanced Accuracy Score', bas], ['Fbeta Score', fbs], 
        ['Hamming Loss', hl], ['Zero One Loss', zol]]
dfResult = pd.DataFrame(data, columns=['Metric', 'Value'])
dfResult.to_csv('RandomForestResult.csv')




""" #################### Graph ###################### """
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)
principalComponentsTest = pca.transform(X_test)


h = 0.01

# Create color maps
cmap_light = ListedColormap(['#FFAAAA','#00AAFF','#90EE90', '#FFFF66'])
cmap_bold0 = ListedColormap(['#FF0000']) 
cmap_bold1 = ListedColormap(['#0000FF'])
cmap_bold2 = ListedColormap(['#0d865d'])
cmap_bold3 = ListedColormap(['#ffde3a'])

# we create an instance of Neighbours Classifier and fit the data.


# calculate min, max and limits
x_min, x_max = math.floor(min(principalComponentsTest[:,0].min(), 
                              principalComponents[:,0].min())), math.ceil(max(principalComponentsTest[:,0].max(), 
                                                                              principalComponents[:,0].max()))
y_min, y_max =math.floor(min(principalComponentsTest[:,1].min(), 
                             principalComponents[:,1].min())), math.ceil(max(principalComponentsTest[:,1].max(), 
                                                                             principalComponents[:,1].max()))

minmaxx = np.arange(x_min, x_max, h)
minmaxy = np.arange(y_min, y_max, h)
xx, yy = np.meshgrid(minmaxx,minmaxy)

clf = RandomForestClassifier(n_jobs = -1)
clf.fit(principalComponents, y_train)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

p0 = [[principalComponentsTest[x,0], principalComponentsTest[x,1]] 
      for x in range(len(y_test)) if y_test[x]==0]
p0 = np.array(p0)
c0 = np.zeros(5054)
p1 = [[principalComponentsTest[x,0], principalComponentsTest[x,1]] 
      for x in range(len(y_test)) if y_test[x]==1]
p1 = np.array(p1)
c1 = np.ones(4612)
p2 = [[principalComponentsTest[x,0], principalComponentsTest[x,1]] 
      for x in range(len(y_test)) if y_test[x]==2]
p2 = np.array(p2)
c2 = np.ones(3543)+1
p3 = [[principalComponentsTest[x,0], principalComponentsTest[x,1]] 
      for x in range(len(y_test)) if y_test[x]==3]
p3 = np.array(p3)
c3 = np.ones(2428)+2


# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


a = plt.scatter(p0[:, 0], p0[:, 1], c=c0, cmap=cmap_bold0, label = 'class 0')   
b = plt.scatter(p1[:, 0], p1[:, 1], c=c1, cmap=cmap_bold1, label = 'class 1')
c = plt.scatter(p2[:, 0], p2[:, 1], c=c2, cmap=cmap_bold2, label = 'class 2')
d = plt.scatter(p3[:, 0], p3[:, 1], c=c3, cmap=cmap_bold3, label = 'class 3')
plt.axis(xmin=xx.min(),xmax=xx.max())
plt.axis(ymin=yy.min(), ymax=yy.max())
plt.legend(handles = [a,b,c,d])

