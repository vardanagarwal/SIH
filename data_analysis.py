# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:32:15 2020

@author: hp
"""

import pandas as pd

df = pd.read_csv('meta.csv')

#%% which rows of the last columns have missing values

df1 = df.iloc[:, 47:]
df1.insert(0, 'Filename', df.iloc[:, 0])
df1.insert(1, 'CamId', df.iloc[:, 1])
df2 = df1[df1.isna().any(axis=1)]
top = df2.head(10)

#%%correlation between metadata

lst = list(df)
df1 = pd.concat([df.iloc[:, 12:24], df.iloc[:, 40:45], df.iloc[:, 47:]], axis=1)
top = df1.head(10)
df2 = df1.dropna()
pearson = df2.corr()
spearman = df2.corr('spearman')