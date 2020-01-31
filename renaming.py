# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:18:43 2020

@author: hp
"""

# import cv2
import os
import re

images = os.listdir('first\\training')
os.makedirs('first\\')
total = len(images)
for i in range(total):
    new = str(i)+'.'+re.split('\\.', images[i])[1]
    os.rename('first\\training\\'+images[i], 'first\\training\\'+new)
    