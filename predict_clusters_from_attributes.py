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

from sklearn.metrics import classification_report, confusion_matrix
#%%
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# [[9595   71]
#  [  56 5915]]
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      9666
#            1       0.99      0.99      0.99      5971

#     accuracy                           0.99     15637
#    macro avg       0.99      0.99      0.99     15637
# weighted avg       0.99      0.99      0.99     15637

#%%
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
 learning_rate=0.1,
 n_estimators=1000,
 max_depth=11,
 min_child_weight=1,
 gamma=0,
 subsample=0.5,
 colsample_bytree=0.5,
 nthread=4,
  objective= 'binary:logistic',
 scale_pos_weight=1.6,
 seed=27,
 random_state=42,
 verbosity=0)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# [[9607   59]
#  [  58 5912]]
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      9666
#            1       0.99      0.99      0.99      5971

#     accuracy                           0.99     15637
#    macro avg       0.99      0.99      0.99     15637
# weighted avg       0.99      0.99      0.99     15637

#%%
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# [[9591   75]
#  [  98 5873]]
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      9666
#            1       0.99      0.98      0.99      5971

#     accuracy                           0.99     15637
#    macro avg       0.99      0.99      0.99     15637
# weighted avg       0.99      0.99      0.99     15637

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 40))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 32, epochs = 150, validation_data=(X_test, y_test))
y_pred = classifier.predict_classes(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# [[9582   84]
#  [  47 5924]]
#               precision    recall  f1-score   support

#            0       1.00      0.99      0.99      9666
#            1       0.99      0.99      0.99      5971

#     accuracy                           0.99     15637
#    macro avg       0.99      0.99      0.99     15637
# weighted avg       0.99      0.99      0.99     15637
#%%
from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


X = pd.concat([df1.iloc[:, 2:-1], df2.iloc[:, 2:-1]])
y = pd.concat([df1.iloc[:, -1], df2.iloc[:, -1]])

pca = make_pipeline(StandardScaler(),
                    PCA(n_components=2, random_state=0))
# lda = make_pipeline(StandardScaler(),
#                     LinearDiscriminantAnalysis(n_components=2))
knn = KNeighborsClassifier(n_neighbors=5)
dim_reduction_methods = [('PCA', pca)]#, ('LDA', lda)]

for i, (name, model) in enumerate(dim_reduction_methods):
    plt.figure()
    # plt.subplot(1, 3, i + 1, aspect=1)

    # Fit the method's model
    model.fit(X_train, y_train)

    # Fit a nearest neighbor classifier on the embedded training set
    knn.fit(model.transform(X_train), y_train)

    # Compute the nearest neighbor accuracy on the embedded test set
    acc_knn = knn.score(model.transform(X_test), y_test)

    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = model.transform(X)

    # Plot the projected points and show the evaluation score
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
    plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name,
                                                              5,
                                                              acc_knn))
plt.show()

                                                   
