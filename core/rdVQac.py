from core.cwSaab import cwSaab
from core.util.myKMeans import myKMeans
from core.util.Huffman import Huffman
import numpy as np
from core.util import Time, myLog, Shrink
from core.util.ac import HierarchyCABAC
from core.util.evaluate import MSE
from core.util.ReSample import *
from core.VQEntropy import VQEntropy
myLog('<FRAMEWORK> rdVQ 2022.12.09')

def toSpatial(cwSaab, iR, level, S,tX):
    for i in range(level, -1, -1):
        if i > 0:
            iR = cwSaab.inverse_transform_one(iR, tX[i-1], i)
        else:
            iR = cwSaab.inverse_transform_one(iR, None, i)
    return iR

class VQ:
    def __init__(self, n_clusters_list, win_list, n_dim_list, enable_skip={}, transform_split=0,Lagrange_multip=300000):
        self.n_clusters_list = n_clusters_list
        self.win_list = win_list
        self.n_dim_list = n_dim_list
        self.cwSaab = cwSaab(win=win_list, TH=-1, transform_split=transform_split)
        self.shape = {}
        self.myKMeans = {}
        self.Huffman = {}
        self.buffer = {}
        self.acc_bpp = 0
        self.Lagrange_multip = Lagrange_multip
        self.fast=True

    def to_spatial(self,iR, tX, level, useTx=False):
        for i in range(level, -1, -1):
            if i > 0:
                if useTx == True:
                    iR = self.cwSaab.inverse_transform_one(iR, tX[i-1], i)
                else:
                    iR = self.cwSaab.inverse_transform_one(iR, np.zeros_like(tX[i-1]), i)
            else:
                iR = self.cwSaab.inverse_transform_one(iR, None, i)
        return iR
    
    def RD_search(self, dmse, mse, omse, pidx, label, myhash, S, fit=False):
        min_cost, th = 1e40, -1
        mr, md = 0, 0
        lcost = 1e40
        step = 100
        if fit == False and 200 not in self.Huffman[myhash].keys():
            step = 400
        for skip_TH in range(0, 80000, step):
            idx = (dmse > skip_TH).reshape(-1)
            if fit == True or self.fast==True:
                st0=''
                l = 2.75*np.sum(idx)/len(idx)+0.00276
                l = int(l*len(idx))
                for i in range(l+1):
                    st0+='0'
            else:
                st0 = HierarchyCABAC().encode(pidx, idx.reshape(S), 1) 
            if self.Huffman[myhash][skip_TH] is None:
                st0, st1 = '', ''
                idx = np.zeros_like(idx)
            else:
                st1 = self.Huffman[myhash][skip_TH].encode(label, idx.reshape(S))
            r = len(st0+st1) / S[0] / 1024**2
            d = np.zeros_like(mse)
            d[idx==True] = mse[idx==True]
            d[idx==False] = omse[idx==False]
            d = np.mean(d)
            cost = d + self.Lagrange_multip * r * pow(1.1, 8-int(myhash[1]))
            if min_cost > cost:
                min_cost = cost
                th = skip_TH
                mr, md = r, d
            if lcost < cost:
                break
            else:
                lcost = cost
            
        return th, [min_cost, mr, md]

    def fit_huffmans(self, myhash, dmse, label, S):
        h = {}
        nc = self.n_clusters_list[int(myhash[1])][int(myhash[-1])]
        for skip_TH in range(0, 80000, 100):
            idx = dmse > skip_TH
            if np.sum(idx) <1:
                h[skip_TH]=None
                continue
            h[skip_TH] = VQEntropy(nc, self.myKMeans[myhash].inverse_predict(np.arange(nc).reshape(-1, 1))).fit(label.reshape(S), idx.reshape(S))
        self.Huffman[myhash] = h
        
    @Time
    def select(self, tX, X, iX, label, level, pos, pidx=None, isfit=False):
        myhash = 'L'+str(level)+'-P'+str(pos)
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        siX = np.zeros_like(X)
        siX[:, :,:,:self.n_dim_list[level][pos]] += iX
        if self.fast == True:#self.fast:
            sX = X
        else:
            sX, siX = self.to_spatial(X, tX, level, True), self.to_spatial(siX, tX, level)
            acc_win = 1
            for i in range(0, level+1):
                acc_win *= self.win_list[i]
            sX, siX = Shrink(sX, acc_win), Shrink(siX, acc_win)
        sX, siX = sX.reshape(-1, sX.shape[-1]), siX.reshape(-1, siX.shape[-1])
        mse = (np.mean(np.square(sX-siX),axis=1))
        omse =  np.mean(np.square(sX), axis=1)
        sX, siX = 0, 0
        dmse = omse-mse
        if isfit == True:
            self.fit_huffmans(myhash, dmse, label, S)
        th, cost = self.RD_search(dmse, mse, omse, pidx, label, myhash, S, isfit)
        idx = (dmse > th).reshape(-1)
        iX = iX.reshape(-1, iX.shape[-1])
        iX[idx == False] *= 0
        iX, idx = iX.reshape(S), idx.reshape(S)
        st0 = HierarchyCABAC().encode(pidx, idx, 1)
        if self.Huffman[myhash][th] is None:
            st1 = ''
            st0 = ''
            iX *= 0
        else:
            st1 = self.Huffman[myhash][th].encode(label.reshape(S),idx.reshape(S))
        myLog('<INFO> RD_cost=%8.4f r=%f d=%4.5f Skip_TH=%d'%(cost[0], cost[1], cost[2], th))
        myLog('<BITSTREAM> st_idx=%d'%len(st0))
        myLog('<BITSTREAM> st=%d'%len(st1))
#         self.buffer[myhash+'_idx'] = idx
#         self.buffer[myhash+'_label_st'] = st1
#         self.buffer[myhash+'_idx_st'] = st0
#         self.buffer[myhash+'_th'] = th
        return iX
        
    @Time
    def fit_one_level_one_pos(self, X, tX, level, pos):
        myhash = 'L'+str(level)+'-P'+str(pos)
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], X.shape[-1])
        myLog('id=%s vq_dim=%d n_clusters=%d'%(myhash, self.n_dim_list[level][pos], self.n_clusters_list[level][pos]))
        self.myKMeans[myhash] = myKMeans(self.n_clusters_list[level][pos]).fit(X[:, :,:,:self.n_dim_list[level][pos]])
        label = self.myKMeans[myhash].predict(X[:,:,:, :self.n_dim_list[level][pos]])        
        iX = self.myKMeans[myhash].inverse_predict(label)
        iX = self.select(tX, X, iX, label, level, pos, self.buffer.get('L'+str(level+1)+'-P'+str(0)+'_idx', None), True)
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
        iX = self.select(tX, X, iX, label, level, pos, self.buffer.get('L'+str(level+1)+'-P'+str(0)+'_idx', None))
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

    def predict(self, X, skip=[],fast=False):
        print('rdvq', 1.1)
        self.fast = fast
        if X.shape[0] > 300:
            self.fast=True
        else:
            self.fast=False
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

    