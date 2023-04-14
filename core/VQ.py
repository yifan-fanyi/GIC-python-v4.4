from core.cwSaab import cwSaab
from core.util.myKMeans import myKMeans
from core.util.Huffman import Huffman
import numpy as np
from core.util import Time, myLog, Shrink
from core.util.ac import HierarchyCABAC
from core.util.evaluate import MSE
from core.util.ReSample import *


def toSpatial(cwSaab, iR, level, S,tX):
    for i in range(level, -1, -1):
        if i > 0:
            s = S[i-1]
            iR = cwSaab.inverse_transform_one(iR, tX[i-1], i)
        else:
            iR = cwSaab.inverse_transform_one(iR, None, i)
    return iR

class VQ:
    def __init__(self, n_clusters_list, win_list, n_dim_list, enable_skip={}, transform_split=0):
        self.n_clusters_list = n_clusters_list
        self.win_list = win_list
        self.n_dim_list = n_dim_list
        self.cwSaab = cwSaab(win=win_list, TH=-1, transform_split=transform_split)
        self.shape = {}
        self.myKMeans = {}
        self.Huffman = {}
        self.enable_skip = enable_skip
        self.buffer = {}

    def to_spatial(self,iR, tX, level):
        for i in range(level, -1, -1):
            if i > 0:
                iR = self.cwSaab.inverse_transform_one(iR, np.zeros_like(tX[i-1]), i)
            else:
                iR = self.cwSaab.inverse_transform_one(iR, None, i)
        return iR
    
    @Time
    def select(self, tX, X, iX, skip_TH, level, pos, pidx=None):
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        siX = np.zeros_like(X)
        siX[:, :,:,:self.n_dim_list[level][pos]] += iX
        sX, siX = self.to_spatial(X, tX, level), self.to_spatial(siX, tX, level)
        acc_win = 1
        for i in range(0, level+1):
            acc_win *= self.win_list[i]
        sX, siX = Shrink(sX, acc_win), Shrink(siX, acc_win)
        sX, siX = sX.reshape(-1, sX.shape[-1]), siX.reshape(-1, siX.shape[-1])
        idx = (np.mean(np.square(sX), axis=1)-np.mean(np.square(sX-siX),axis=1)) > skip_TH
        iX = iX.reshape(-1, iX.shape[-1])
        iX[idx.reshape(-1) == False] *= 0
        iX, idx = iX.reshape(S), idx.reshape(S)
        if iX.shape[0] < 200:
            ac = HierarchyCABAC()
            myLog('<BITSTREAM> idx %d'%len(ac.encode(pidx, idx, 1)))
        return idx, iX
        
    @Time
    def fit_one_level_one_pos(self, X, tX, level, pos):
        myhash = 'L'+str(level)+'-P'+str(pos)
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], X.shape[-1])
        myLog('id=%s vq_dim=%d n_clusters=%d'%(myhash, self.n_dim_list[level][pos], self.n_clusters_list[level][pos]))
        self.myKMeans[myhash] = myKMeans(self.n_clusters_list[level][pos]).fit(X[:, :,:,:self.n_dim_list[level][pos]])
        label = self.myKMeans[myhash].predict(X[:,:,:, :self.n_dim_list[level][pos]])        
        iX = self.myKMeans[myhash].inverse_predict(label)
        if self.enable_skip.get(myhash, [False,0])[0] == True:
            idx, iX = self.select(tX, X, iX, self.enable_skip.get(myhash, [False,0])[1], level, pos, self.buffer.get('L'+str(level+1)+'-P'+str(0)+'_idx', None))
            self.buffer[myhash+'_idx'] = idx
            if np.sum(idx) > 1:
                self.Huffman[myhash] = Huffman().fit(label.reshape(-1)[idx.reshape(-1)])
            else:
                self.Huffman[myhash] = None
        else:
            self.Huffman[myhash] = Huffman().fit(label)
        X[:, :,:,:self.n_dim_list[level][pos]] -= iX
        return X
    
    @Time
    def fit_one_level(self, iR, tX, level):
        myhash = 'L'+str(level)
        self.shape[myhash] = [iR.shape[0], iR.shape[1], iR.shape[2], -1]
        myLog('id=%s'%myhash)
        for pos in range(len(self.n_dim_list[level])):
            iR = self.fit_one_level_one_pos(iR, tX, level, pos)
        return iR.reshape(self.shape[myhash])

    @Time
    def fit(self, X):
        self.cwSaab.fit(X)
        tX = self.cwSaab.transform(X)
        iR = tX[-1]
        for level in range(len(self.n_dim_list)-1, -1, -1):
            iR = self.fit_one_level(iR, tX, level)
            if level > 0:
                iR = self.cwSaab.inverse_transform_one(iR, tX[level-1], level)
            else:
                iR = self.cwSaab.inverse_transform_one(iR, None, level)
        return iR

    def predict_one_level_one_pos(self, tX, X, level, pos, skip):
        myhash = 'L'+str(level)+'-P'+str(pos)
        myLog('id=%s'%(myhash))
        if myhash in skip:
            myLog('<INFO> SKIP')
            return X
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], X.shape[-1])
        label = self.myKMeans[myhash].predict(X[:, :,:,:self.n_dim_list[level][pos]])
        iX = self.myKMeans[myhash].inverse_predict(label)
        if self.enable_skip.get(myhash, [False,0])[0] == True:
            idx, iX = self.select(tX, X, iX, self.enable_skip.get(myhash, [False,0])[1], level, pos, self.buffer.get('L'+str(level+1)+'-P'+str(0)+'_idx', None))
            self.buffer[myhash+'_idx'] = idx
            try:
                st = self.Huffman[myhash].encode(label.reshape(-1)[idx.reshape(-1)])
            except:
                st = ''
        else:
            st = self.Huffman[myhash].encode(label)
        myLog('<BITSTREAM>%s st %d'%(myhash, len(st)))
        X[:,:,:, :self.n_dim_list[level][pos]] -= iX
        tx =  toSpatial(self.cwSaab, X.copy(), level, self.S, self.rX)
        myLog('<INFO> %s local MSE=%4.3f MAD=%4.3f'%(myhash, MSE(np.zeros_like(tx), tx), np.mean(np.abs(tx))))
        return X
    
    #@Time
    def predict_one_level(self, tX, iR, level, skip):
        myhash = 'L'+str(level)
        self.shape[myhash] = [iR.shape[0], iR.shape[1], iR.shape[2], -1]
        for pos in range(len(self.n_dim_list[level])):
            iR = self.predict_one_level_one_pos(tX, iR, level, pos, skip)
        return iR.reshape(self.shape[myhash])

    def predict(self, X, skip=[]):
        self.buffer = {}
        self.S = []
        tX = self.cwSaab.transform(X)
        self.rX = tX.copy()
        for i in tX:
            self.S.append(i.shape)
        iR = tX[-1]
        for level in range(len(self.n_dim_list)-1, -1, -1):
            iR = self.predict_one_level(tX, iR, level, skip)
            if level > 0:
                iR = self.cwSaab.inverse_transform_one(iR, tX[level-1], level)
            else:
                iR = self.cwSaab.inverse_transform_one(iR, None, level)            
        return iR

    