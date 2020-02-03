# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:31:56 2020

@author: hp
"""

import cv2
import os

masks = os.listdir('mask')
os.makedirs('mask_01')
for mask in masks:    
    img = cv2.imread('mask\\'+mask)
    _, thresh = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
    cv2.imwrite('mask_01\\'+mask, thresh)