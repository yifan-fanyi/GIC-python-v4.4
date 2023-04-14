# 2021.10
# @yifan
import numpy as np
from PIL import Image

class ReSample:
    @staticmethod
    def resize(I, S):
        if I.shape[1] < S:
            while I.shape[1] < S:
                I = ReSample.resample(I, 1/2)
        else:
            while I.shape[1] > S:
                I = ReSample.resample(I, 2)
        return I
        
    @staticmethod
    def split(X, ratio=2, mode='LANCZOS'):
        DC_L = ReSample.resample(X, ratio, mode)
        X_inv_L = ReSample.resample(DC_L, 1/ratio, mode)
        AC = X - X_inv_L
        return DC_L, AC
        
    @staticmethod
    def inv_split(DC, AC, ratio=2, mode='LANCZOS'):
        DC = ReSample.resample(DC, 1/ratio, mode)
        return DC + AC

    @staticmethod
    def resample(X, ratio=2, mode='LANCZOS'):
        image_list = []
        for i in range(X.shape[0]):
            tmp = []
            for k in range(X.shape[-1]):
                size1 = int(X.shape[1]/ratio)
                size2 = int(X.shape[2]/ratio)
                if mode == "NEAREST" or mode == 0:
                    image_tmp = Image.fromarray(np.float32(X[i,:,:,k])).resize(size=(size1, size2), resample=Image.NEAREST)
                elif mode == "BILINEAR" or mode == 1:
                    image_tmp = Image.fromarray(np.float32(X[i,:,:,k])).resize(size=(size1, size2), resample=Image.BILINEAR)
                elif mode == "BICUBIC" or mode == 2:
                    image_tmp = Image.fromarray(np.float32(X[i,:,:,k])).resize(size=(size1, size2), resample=Image.BICUBIC)
                else:
                    image_tmp = Image.fromarray(np.float32(X[i,:,:,k])).resize(size=(size1, size2), resample=Image.LANCZOS)
                tmp.append(np.float16(image_tmp).reshape(1, size1, size2, 1))
            image_list.append(np.concatenate(tmp, axis=-1))
        output = np.concatenate(image_list, axis=0)
        #output[output < 0] = 0
        #output[output > 255] = 255
        return np.round(output)

def resize(Y, S):
    while Y.shape[2] > S:
        Y = ReSample.resample(Y,2)
    while Y.shape[2] < S:
        Y = ReSample.resample(Y,1/2)
    return Y
