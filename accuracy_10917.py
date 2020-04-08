# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:15:56 2020

@author: hp
"""

import tensorflow as tf
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

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

sample_msk = os.path.join('mask', str(10917)+'.png')
sample_mask = cv2.resize(cv2.imread(sample_msk, 0), (224,224))


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def usable_mask(pred_mask):
    pred_mask = np.array(pred_mask, dtype=np.uint8)
    pred_mask = np.reshape(pred_mask, (224, 224))
    return pred_mask

def cal_acc(pred_mask, sample_mask):
    incorrect = cv2.bitwise_xor(pred_mask, sample_mask)
    return (224*224 - np.sum(incorrect))/ (224*224)

def false_pos(pred_mask, sample_mask):
    sample_mask = cv2.bitwise_not(sample_mask)
    val = cv2.bitwise_and(sample_mask, pred_mask)
    return (224*224 - np.sum(val))/ (224*224)

def false_neg(pred_mask, sample_mask):
    pred_mask = cv2.bitwise_not(pred_mask)
    val = cv2.bitwise_and(pred_mask, sample_mask)
    return (224*224 - np.sum(val))/ (224*224)

def find_acc(label, sample_img):
    
    global model
    global sample_mask
    acc = [0]*3
    sample_image = cv2.resize(cv2.cvtColor(cv2.imread(sample_img), cv2.COLOR_BGR2RGB), (224,224))/255.
    pred_mask = create_mask(model[label].predict(sample_image[tf.newaxis, ...]))
    pred_mask1 = usable_mask(pred_mask)
    acc[0] = cal_acc(pred_mask1, sample_mask)
    acc[1] = false_pos(pred_mask1, sample_mask)
    acc[2] = false_neg(pred_mask1, sample_mask)
    return acc

meta = pd.read_csv('meta.csv')
meta = meta[meta.CamId == 10917]
meta = meta.iloc[:, :9]
meta = meta.drop(columns=['Year', 'hoiem_MCR', 'svetlana2010_MCR', 'tcwc_MCR'])
meta = meta.reset_index(drop=True)
cluster_10917 = pd.read_csv('cluster_10917.csv')
acc = []
for i in tqdm(range(len(cluster_10917))):
    sample_img = os.path.join('dataset', str(10917), str(10917), cluster_10917.Filename[i])
    acc.append(find_acc(cluster_10917.labels[i], sample_img))
    
acc = pd.DataFrame(acc, columns=['bycluster', 'bycluster_falsepos', 'bycluster_falseneg'])
cluster = pd.concat([meta, acc], axis=1)
cluster.to_csv('predicted_accuracies_10917.csv', index=0)