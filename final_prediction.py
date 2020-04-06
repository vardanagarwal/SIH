# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:48:15 2020

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
         load_model('Model_1.json', 'modelweights_1.h5'),
         load_model('Model.json', 'modelweights.h5')]

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

def find_acc(label, sample_img, sample_msk):
    
    global model
    acc = [0]*12
    sample_image = cv2.resize(cv2.cvtColor(cv2.imread(sample_img), cv2.COLOR_BGR2RGB), (224,224))/255.
    sample_mask = cv2.resize(cv2.imread(sample_msk, 0), (224,224))
    pred_mask = create_mask(model[label].predict(sample_image[tf.newaxis, ...]))
    pred_mask1 = usable_mask(pred_mask)
    acc[0] = cal_acc(pred_mask1, sample_mask)
    acc[1] = false_pos(pred_mask1, sample_mask)
    acc[2] = false_neg(pred_mask1, sample_mask)
    
    pred_mask = create_mask(model[2].predict(sample_image[tf.newaxis, ...]))
    pred_mask2 = usable_mask(pred_mask)
    acc[3] = cal_acc(pred_mask2, sample_mask)
    acc[4] = false_pos(pred_mask2, sample_mask)
    acc[5] = false_neg(pred_mask2, sample_mask)
    
    pred_mask = cv2.bitwise_and(pred_mask1, pred_mask2)
    acc[6] = cal_acc(pred_mask, sample_mask)
    acc[7] = false_pos(pred_mask, sample_mask)
    acc[8] = false_neg(pred_mask, sample_mask)
    
    pred_mask = cv2.bitwise_or(pred_mask1, pred_mask2)
    acc[9] = cal_acc(pred_mask, sample_mask)
    acc[10] = false_pos(pred_mask, sample_mask)
    acc[11] = false_neg(pred_mask, sample_mask)
    
    return acc
        
meta = pd.read_csv('meta.csv')
camids = [75, 1093, 5021, 19834, 8438]
df1 = meta[meta['CamId'].isin(camids)]
df = pd.read_csv('Final_test_set.csv')
s1 = pd.merge(df, df1, how='inner', on=['Filename', 'CamId'])
s1 = s1.iloc[:, :10]
s1 = s1.drop(columns=['Year', 'hoiem_MCR', 'svetlana2010_MCR', 'tcwc_MCR'])
acc = []
for i in tqdm(range(len(s1))):
    sample_img = os.path.join('dataset', str(s1.CamId[i]), str(s1.CamId[i]), s1.Filename[i])
    sample_msk = os.path.join('mask', str(s1.CamId[i])+'.png')
    acc.append(find_acc(s1.label[i], sample_img, sample_msk))
    
acc = pd.DataFrame(acc, columns=['bycluster', 'bycluster_falsepos', 'bycluster_falseneg',
                                 'combined', 'combined_falsepos', 'combined_falseneg',
                                 'and', 'and_falsepos', 'and_falseneg', 
                                 'or', 'or_falsepos', 'or_falseneg'])
final_df = pd.concat([s1, acc], axis=1)
final_df.to_csv('predicted_accuracies.csv', index=0)

