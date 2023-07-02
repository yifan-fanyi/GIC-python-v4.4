from core.rdvq1ac import VQ
import numpy as np
from core.util import *
from core.util.evaluate import MSE
import json
import os
from core.util.ReSample import resize
# .iR residual, will be keep updated
from multiprocessing import Process
import cv2

X = []
for i in range(144):
    x = cv2.imread('/Users/alex/Desktop/proj/data/kodak_256/'+str(i)+'.png')
    X.append(x)
X = np.array(X)
print(X.shape)
r = [8,16,32,64,128,256]
for i in r:
    write_pkl('/Users/alex/Desktop/proj/data/test/'+str(i)+'.spatial_raw', resize(X, i))
