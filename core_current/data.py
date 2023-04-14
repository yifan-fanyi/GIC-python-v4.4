import numpy as np
from core.util.ReSample import resize
from core.util.load_img import *
import pickle
import os
from core.util import Time

if os.getlogin() == 'alex':
    CLIC = '/Users/alex/Desktop/proj/data/'
else:
    Imagenet_pkl = '/mnt/zhanxuan/data/ImageNet_Our/train/'
    CLIC_pkl = '/mnt/yifan/data/Ours/'
    CLIC = '/mnt/yifan/data/CLIC/'
    Holopix50k = "/mnt/yifan/data/Holopix50k/"



@Time
def load_imagenet(ct):
    Y8, Y32 = [], []
    arr = np.arange(1000)
    np.random.shuffle(arr)
    for ii in range(1000):
        i = arr[ii]
        try:
            with open(Imagenet_pkl+str(8)+'/'+str(i)+'.pkl','rb') as f:
                d8 = pickle.load(f)
            with open(Imagenet_pkl+str(32)+'/'+str(i)+'.pkl','rb') as f:
                d32 = pickle.load(f)
            if d8.shape[1] != 8:
                continue
            if d32.shape[1] != 32:
                continue
        except:
            continue
        Y8.append(d8)
        Y32.append(d32)
        c += 1
        if ct > 0:
            break
    return np.concatenate(Y8, axis=0), np.concatenate(Y32, axis=0)

@Time
def load_(mode, size, ct=100):
    Y = []
    arr = np.arange(199)
    np.random.shuffle(arr)
    for ii in range(ct):
        i = arr[ii]
        with open(CLIC_pkl+mode+'/'+str(size)+'/'+str(i)+'.pkl', 'rb') as f:
            d = pickle.load(f)
        Y.append(d)
    return np.concatenate(Y, axis=0)

def load(Rtype, ct, size=[1024,256,32,8]):
    if Rtype == 'small':
        Y80, Y320 = load_('train', 8, ct=190), load_('train', 32, ct=ct[0])
        Y81, Y321 = load_imagenet(ct=ct[1])
        Y8 = np.concatenate([Y80,Y81], axis=0)
        Y32 = np.concatenate([Y320,Y321], axis=0)
        return Y8, Y32
    if Rtype == 'test':
        Yt = Load_from_Folder(folder=CLIC+"test"+str(size[0])+"/", color='RGB', ct=-1)
        Yt = np.array(Yt).astype('float16')
    if Rtype == 'train':
        Yt = Load_from_Folder(folder=CLIC+"train"+str(size[0])+"/", color='RGB', ct=-1)
        try:
            a = Load_from_Folder(folder=Holopix50k+'train'+str(size[0])+'/', color='RGB', ct=ct)
            Yt += a
        except:
            pass
        Yt = np.array(Yt).astype('float16')
    Y = [Yt]
    for i in range(1, len(size)):
        Y.append(resize(Y[-1], size[i]))
    return Y
