# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:06:48 2020

@author: hp
"""
#moving files that are present in extra folder to base like those not present

import shutil
import os

pth = 'C:\\Users\\hp\\Documents\\SIH'
os.chdir(pth)
folders = os.listdir('dataset')
pth = os.path.join(pth, 'dataset')
os.chdir(pth)
extra = 'home\\mihail\\mypages\\rpmihail\\skyfinder\\images'
for folder in folders:
    name = os.listdir(folder)
    if name[0] != folder:
        src = os.path.join(pth, folder, extra, folder)
        dst = os.path.join(pth, folder, folder)
        os.makedirs(dst)
        files = os.listdir(src)
        for file in files:
            shutil.move(src+"\\"+file, dst)

#%%removing the extra empty folders
import os
import shutil

pth = 'C:\\Users\\hp\\Documents\\SIH\\dataset'
folders = os.listdir((pth))
for folder in folders:
    files = os.listdir(folder)
    for file in files:
        if file == 'home':
            temp = os.path.join(pth, folder, file)
            shutil.rmtree(temp)
