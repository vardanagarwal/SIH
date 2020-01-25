# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:58:07 2020

@author: hp
"""

import os
import shutil
import re

pth = 'C:\\Users\\hp\\Documents\\SIH\\mask'
dst = 'C:\\Users\\hp\\Documents\\SIH\\dataset'
masks = os.listdir(pth)
for mask in masks:
    val = re.split('\.', mask)[0]
    shutil.move(pth+'\\'+mask, dst+'\\'+val)
