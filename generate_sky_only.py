# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:16:13 2020

@author: hp
"""

import os
import cv2

pth = 'C:\\Users\\hp\\Documents\\SIH\\dataset'
done = ['10066', '10870', '10917', '1093', '11160', '11331', '162', '17218', '17244', '18590', '19106' ,'19306', '19388']
folders = os.listdir(pth)
for folder in folders:
    if folder not in done:
        dst = os.path.join(pth, folder, "sky")
        os.makedirs(dst)
        src = os.path.join(pth, folder, folder)
        images = os.listdir(src)
        mask_path = os.path.join(pth, folder, folder)+".png"
        mask = cv2.imread(mask_path, 0)
        for image in images:
            img = cv2.imread(src+'\\'+image)
            if mask.shape != img.shape:
                img = cv2.resize(img, (mask.shape[1], mask.shape[0]), interpolation = cv2.INTER_LINEAR)
                cv2.imwrite(src+'\\'+image, img)
            res = cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite(dst+'\\'+image, res)
