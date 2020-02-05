# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:08:44 2020

@author: hp
"""

import cv2
import numpy as np
import pandas as pd
import os

os.chdir('dataset')
def find_value(img, mask):
    res = [0, 0, 0]
    for i in range(3):
        res[i] = np.sum(np.multiply(img[:, :, i], mask))/np.sum(mask)
    return res

result = []
folders = os.listdir()
for folder in folders: 
    print(folder)
    if folder not in ['10917']:
        images = os.listdir(folder + '\\' + folder)
        mask = cv2.imread(folder + '\\' + folder + '.png', 0)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        for image in images:
            img = cv2.imread(folder + '\\'  + folder + '\\'  + image)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            
            b, g, r = np.divide(find_value(img, mask), [255, 255, 255])
            h, s, v = np.divide(find_value(hsv, mask), [180, 255, 255])
            y, cr, cb = np.divide(find_value(ycbcr, mask), [255, 255, 255])
            
            values = [folder, image, r, g, b, h, s, v, y, cr, cb]
            result.append(values)
            
df = pd.DataFrame(result)
df.columns = ['cam_id', 'filename', 'r', 'g', 'b', 'h', 's', 'v', 'y', 'cr', 'cb']
df.to_csv('sky_normalization.csv', index=False)