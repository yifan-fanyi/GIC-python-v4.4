import numpy as np
from core.util.ReSample import resize
from core.util.load_img import *
import pickle
import os
from core.util import Time
import pickle
# with open('name.pkl', 'rb') as f:
#     d = pickle.load(f)
# r = d['data_root']
r=''
Imagenet_pkl = r + 'imagenet'
CLIC_pkl = r+'clic'
CLIC = r+'CLIC/'
Holopix50k = r+"Holopix50k/"
CLIC = '/Users/alex/Desktop/proj/data/'
@Time
def load_imagenet(ct):
    Y8, Y32 = [], []
    ii = np.arange(1000)
    np.random.shuffle(ii)
    for i in ii:
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
        ct += 1
        if ct > 0:
            break
    return np.concatenate(Y8, axis=0), np.concatenate(Y32, axis=0)

@Time
def load_(mode, size, ct=100):
    Y = []                  
    for i in range(ct):
        with open(CLIC_pkl+str(size)+'/'+str(i)+'.pkl', 'rb') as f:
            d = pickle.load(f)
        Y.append(d)
    return np.concatenate(Y, axis=0)

def load(Rtype, ct, size=[1024,256,32,8]):
    if Rtype == 'small':
        Y80, Y320 = load_('train', 8, ct=ct[0]), load_('train', 32, ct=ct[0])
        Y81, Y321 = load_imagenet(ct=ct[1])
        Y8 = np.concatenate([Y80,Y81], axis=0)
        Y32 = np.concatenate([Y320,Y321], axis=0)
        return Y8.astype('float16'), Y32.astype('float16')
    if Rtype == 'test':
        Yt = Load_from_Folder(folder=CLIC+"test"+str(size[0])+"/", color='RGB', ct=-1)
        Yt = np.array(Yt).astype('float16')
    if Rtype == 'train':
        Yt = Load_from_Folder(folder=CLIC+"train"+str(size[0])+"/", color='RGB', ct=ct[0])
        try:
            a = Load_from_Folder(folder=Holopix50k+'train_'+str(size[0])+'/', color='RGB', ct=ct[1])
            Yt += a
        except:
            pass
        Yt = np.array(Yt).astype('float16')
    if Rtype == 'kodak':
        Yt = Load_from_Folder(folder='/home/alex/Documents/kodak_256/', color='RGB', ct=ct[0])
        Yt = np.array(Yt).astype('float16')
    Y = [Yt]
    for i in range(1, len(size)):
        Y.append(resize(Y[-1], size[i]).astype('float16'))
    return Y
