# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 03:36:19 2020

@author: hp
"""

import os
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

os.chdir('0_training\\0_training')
files = os.listdir()
corrupt = []
for img_path in tqdm(files):
    try:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
    except:
        corrupt.append(img_path)
    
os.chdir('C:\\Users\\hp\\Documents\\SIH')
corrupt_training = pd.DataFrame(corrupt, columns=['name'])
corrupt_training.to_csv('corrupt_training_0.csv', index=False)

#%%
os.chdir('0_testing\\0_testing')
files = os.listdir()
corrupt = []
for img_path in tqdm(files):
    try:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
    except:
        corrupt.append(img_path)

os.chdir('C:\\Users\\hp\\Documents\\SIH')
corrupt_testing = pd.DataFrame(corrupt, columns=['name'])
corrupt_testing.to_csv('corrupt_testing_0.csv', index=False)
