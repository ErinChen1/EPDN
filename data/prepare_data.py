import os
import numpy as np
from PIL import Image
import random
# import matplotlib.pyplot as plt
# import cv2
import scipy.io as sio
import glob

def write_txt(i, clearimgs, hazyimgs, train=True):
    basepath = '../datasets/'
    if train:
        path = basepath + 'Train.txt'
    else:
        path = basepath + 'Val.txt'

    with open(path, 'a') as f:
        f.write(str(i) + '\t' + clearimgs + '\t' + hazyimgs + '\n')


ii = 0
trans_data_root='/home/gan/文档/个人文件/实验室/数据集/去雾/ChinaMM/hazy'
trans_gt_root='/home/gan/文档/个人文件/实验室/数据集/去雾/ChinaMM/clear'
for i in range(1, 1300):
    for j in range(1, 11):
        name = '{:d}_{:d}_*.png'.format(i, j)
        pth = glob.glob(os.path.join(trans_data_root, name))[0]
        hazyName=os.path.join(trans_data_root, pth)
        name = '{:d}.png'.format(i)
        pth = glob.glob(os.path.join(trans_gt_root, name))[0]
        clearName=os.path.join(trans_gt_root, pth)
        write_txt(ii, clearName, hazyName, train=True)
        ii+=1


# val_data_root='/home/gan/文档/hjy/dataset/ChinaMM/testdata/indoor/hazy'
# val_gt_root='/home/gan/文档/hjy/dataset/ChinaMM/testdata/indoor/gt'
#
# hazyfiles=os.listdir(val_data_root)
# clearfiles=os.listdir(val_gt_root)
# for i in range(len(hazyfiles)):
#     hazyName = os.path.join(val_data_root, hazyfiles[i])
#     clearName = os.path.join(val_gt_root, clearfiles[i])
#     write_txt(ii, clearName, hazyName, train=False)
#     ii += 1
# for i in range(1301, 1400):
#     for j in range(1, 11):
#         name = '{:d}_{:d}_*.png'.format(i, j)
#         pth = glob.glob(os.path.join(val_data_root, name))[0]
#         hazyName=os.path.join(val_data_root, pth)
#
#         name = '{:d}.png'.format(i)
#         pth = glob.glob(os.path.join(val_gt_root, name))[0]
#         clearName=os.path.join(val_gt_root, pth)
#         write_txt(ii, clearName, hazyName, train=False)
#         ii+=1