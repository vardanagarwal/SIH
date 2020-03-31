# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:47:52 2020

@author: hp
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm

# os.makedirs('0_training')
# os.makedirs('0_testing')
# os.makedirs('1_training')
# os.makedirs('1_testing') 


train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('predicted_clusters.csv')
sets = [test_set]#[train_set, test_set]
set_name = '_testing'

for i in sets:
    for j in tqdm(range(len(i))):
        src = os.path.join('dataset', str(i.loc[j, 'CamId']), str(i.loc[j, 'CamId']), i.loc[j, 'Filename'])
        img_name = str(i.loc[j, 'CamId'])+'-'+i.loc[j, 'Filename']
        dst = os.path.join((str(i.loc[j, 'labels'])+set_name), img_name)
        shutil.copy(src, dst)
    set_name = '_testing'
    