# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:20:45 2020

@author: hp
"""

import pandas as pd

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

X_train = df1.iloc[:, 2:-1].values
X_test = df2.iloc[:, 2:-1].values
y_train = df1.iloc[:, -1].values
y_test = df2.iloc[:, -1].values

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# [[4568  484    0    2]
#  [ 619 3927   35   31]
#  [   0   43 3310  190]
#  [   1   22  126 2279]]
#               precision    recall  f1-score   support

#            0       0.88      0.90      0.89      5054
#            1       0.88      0.85      0.86      4612
#            2       0.95      0.93      0.94      3543
#            3       0.91      0.94      0.92      2428

#     accuracy                           0.90     15637
#    macro avg       0.91      0.91      0.91     15637
# weighted avg       0.90      0.90      0.90     15637