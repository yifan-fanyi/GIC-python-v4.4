from core.cwSaab import cwSaab
import numpy as np
from core.util import *

class IWSaab:
    def __init__(self, win, TH, transform_split):
        self.win = win
        self.TH = TH
        self.transform_split = transform_split
        
    def fit(self, X):
        return self.transform(X)
    @Time
    def transform(self, X):
        m = []
        tX = []
        for i in range(X.shape[0]):
            cw = cwSaab(self.win, self.TH, self.transform_split).fit(X[i:i+1])
            m.append(cw)
            tX.append(cw.transform(X[i:i+1]))
        tmp = []
        for i in range(len(tX[0])):
            tmp.append([])
        for i in range(len(tX)):
            for j in range(len(tX[i])):
                tmp[j].append(tX[i][j])
        for j in range(len(tX[0])):
            tmp[j] = np.concatenate(tmp[j], axis=0)
        return tmp, m
                
    def inverse_transform_one(self, model, X, tX, level):
        iR = []
        for k in range(len(model)):
            if level > 0:
                iR.append(model[k].inverse_transform_one(X[k:k+1], 
                                                        tX[level-1][k:k+1], 
                                                        level))
            else:
                iR.append(model[k].inverse_transform_one(X[k:k+1], None, level))
        return np.concatenate(iR, axis=0)
    def inverse_transform(self, model, tX):
        ix = tX[-1]
        for i in range(len(tX)-1, -1,-1):
            ix = self.inverse_transform_one(model, ix, tX, i)
        return ix

