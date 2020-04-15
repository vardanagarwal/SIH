# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:41:48 2020

@author: hp
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
import os

def load_model(Model_json, Model_h5):
    # Function to load and return neural network model 
    json_file = open(Model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(Model_h5)
    return loaded_model

model = [load_model('Model_0.json', 'modelweights_0.h5'),
         load_model('Model_1.json', 'modelweights_1.h5')]

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def usable_mask(pred_mask):
    pred_mask = np.array(pred_mask, dtype=np.uint8)
    pred_mask = np.reshape(pred_mask, (224, 224))
    return pred_mask

def cal_miou(pred_mask, sample_mask):
    tp = np.sum(cv2.bitwise_and(pred_mask, sample_mask))
    fp = np.sum(cv2.bitwise_and(pred_mask, cv2.bitwise_not(sample_mask)))
    fn = np.sum(cv2.bitwise_and(cv2.bitwise_not(pred_mask), sample_mask))
    return tp/(tp+fp+fn)

def find_miou(label, sample_img, sample_msk):
    global model
    sample_image = cv2.resize(cv2.cvtColor(cv2.imread(sample_img), cv2.COLOR_BGR2RGB), (224,224))/255.
    sample_mask = cv2.resize(cv2.imread(sample_msk, 0), (224,224))
    pred_mask = create_mask(model[label].predict(sample_image[tf.newaxis, ...]))
    pred_mask = usable_mask(pred_mask)
    return(cal_miou(pred_mask, sample_mask))

df = pd.read_csv('predicted_accuracies.csv')
df = df.iloc[:, :-2]
miou = []
for i in tqdm(range(len(df))):
    sample_img = os.path.join('dataset', str(df.CamId[i]), str(df.CamId[i]), df.Filename[i])
    sample_msk = os.path.join('mask', str(df.CamId[i])+'.png')
    miou.append(find_miou(df.label[i], sample_img, sample_msk))
    
miou = pd.DataFrame(miou, columns=['miou'])
final_df = pd.concat([df, miou], axis=1)
final_df.to_csv('predictions.csv')