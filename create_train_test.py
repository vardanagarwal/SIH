# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:05:41 2020

@author: hp
"""

import pandas as pd
from sklearn.model_selection import train_test_split

meta = pd.read_csv('meta.csv')
meta_head = meta.head(10)
cluster = pd.read_csv('kmeans_double_clustered.csv')
cluster_head = cluster.head(10)

df = meta.iloc[:, 47:]
df.insert(0, 'Filename', meta.iloc[:, 0])
df.insert(1, 'CamId', meta.iloc[:, 1])
df1 = df[df.isna().any(axis=1)]
df = df.dropna()
s1 = pd.merge(df, cluster, how='inner', on=['Filename', 'CamId'])


X_train, X_test = train_test_split(s1, test_size=0.2, random_state=42)
X_train.to_csv('train.csv', index=False)
X_test.to_csv('test.csv', index=False)