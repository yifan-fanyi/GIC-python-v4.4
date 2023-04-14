# @yifan
# 2021.01.12
#
import numpy as np
import cv2
import os
from skimage.measure import block_reduce
from core.util.color_space import BGR2RGB, BGR2YUV

def Load_from_Folder(folder, color='RGB', ct=1):
    name = os.listdir(folder)
    np.random.shuffle(name)
    img = []
    Y, U, V = [], [], []
    for n in name:
        try: 
            X = cv2.imread(folder+'/'+n)
            X.shape
        except:
            continue
        if color == 'BGR':
            img.append(X)
        elif color == 'RGB':
            img.append(BGR2RGB(X))
        elif color == 'YUV444' or color == 'YUV':
            Y.append(BGR2YUV(X))
        elif color == 'YUV420':
            X = BGR2YUV(X)
            Y.append(X[:,:,0])
            U.append(block_reduce(X[:,:,1], (2, 2), np.mean))
            V.append(block_reduce(X[:,:,2], (2, 2), np.mean))
        else:
            assert (False), 'No such color type!, Color must be BGR, RGB, YUV, YUV444, or YUV420!'
        ct -= 1
        if ct == 0:
            break
    if color == 'BGR' or color == 'RGB':
        return img
    elif color == 'YUV444' or color == 'YUV':
        return Y
    elif color == 'YUV420':
        return Y, U, V

def Load_Images(name_list, color='RGB'):
    img = []
    Y, U, V = [], [], []
    for n in name_list:
        try: 
            X = cv2.imread(n)
            X.shape   
        except:
            continue
        if color == 'BGR':
            img.append(X)
        elif color == 'RGB':
            img.append(BGR2RGB(X))
        elif color == 'YUV444' or color == 'YUV':
            Y.append(BGR2YUV(X))
        elif color == 'YUV420':
            X = BGR2YUV(X)
            Y.append(X[:,:,0])
            U.append(block_reduce(X[:,:,1], (2, 2), np.mean))
            V.append(block_reduce(X[:,:,2], (2, 2), np.mean))
        else:
            assert (False), 'No such color type!, Color must be BGR, RGB, YUV, YUV444, or YUV420!'
    if color == 'BGR' or color == 'RGB':
        return img
    elif color == 'YUV444' or color == 'YUV':
        return Y
    elif color == 'YUV420':
        return Y, U, V
