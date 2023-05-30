from core.cwSaab import cwSaab
from core.util.myKMeans import myKMeans
from core.util.mydKMeans import mydKMeans

from core.util.Huffman import Huffman
import numpy as np
from core.util import Time, myLog, Shrink, load_pkl, write_pkl
from core.util.ac import HierarchyCABAC, BAC
from core.util.evaluate import MSE
from core.util.ReSample import *
from core.VQEntropy import VQEntropy
import os
isMAD = False
print('<FRAMEWORK> rdVQ1 2022.12.09', isMAD)

def toSpatial(cwSaab, iR, level, S,tX):
    for i in range(level, -1, -1):
        if i > 0:
            iR = cwSaab.inverse_transform_one(iR, tX[i-1], i)
        else:
            iR = cwSaab.inverse_transform_one(iR, None, i)
    return iR

def split_km_subspace(KM, AC):
    def get_dmse(km, AC):    
        label = km.predict(AC)
        iAC = km.inverse_predict(label)
        sX, siX = AC.reshape(-1, AC.shape[-1]), iAC.reshape(-1, iAC.shape[-1])
        mse = (np.mean(np.square((sX-siX).astype('float32')),axis=1))
        omse =  np.mean(np.square(sX.astype('float32')), axis=1)
        dmse = omse-mse
        dmse = dmse.reshape(label.shape)
        return label, iAC, dmse
    label, _, dmse = get_dmse(KM, AC) 
    def labelfilter(l, dmse, nc):
        hist = []
        for i in range(nc):
            hist.append([])
        l = l.reshape(-1)
        dmse = dmse.reshape(-1)
        for i in range(len(l)):
            hist[l[i]].append(dmse[i])
        for i in range(nc):
            hist[i] = np.mean(hist[i])
        return np.argsort(hist)[::-1]
    
    h = labelfilter(label,dmse, len(KM.cluster_centers_))
    cent = KM.inverse_predict(h.reshape(-1,1))
    km_list = []
    nc = 8
    while nc < len(cent):
        km_list.append(myKMeans(-1).fit(X=None, cluster_centers=cent[:nc]))
        nc *=2
    return km_list
class VQ:
    def __init__(self, n_clusters_list, win_list, n_dim_list, enable_skip={}, transform_split=0,Lagrange_multip=300000, acc_bpp=0):
        self.n_clusters_list = n_clusters_list
        self.win_list = win_list
        self.n_dim_list = n_dim_list
        self.cwSaab = cwSaab(win=win_list, TH=-1, transform_split=transform_split)
        self.shape = {}
        self.myKMeans = {}
        self.Huffman = {}
        self.buffer = {}
        self.acc_bpp = acc_bpp
        self.skip_th_range = {}
        self.skip_th_step = {}
        self.Lagrange_multip = Lagrange_multip
        self.fast=True
        self.isdistributed = [-1, None, -1]
        self.max_dmse = {}

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

    # find the optimal threshold
    def RD_search_th(self, myhash, dmse, mse, omse, pidx, label, S, ii, isfit):
        min_cost, th, lcost = 1e40, -1, 1e40
        is0 = False
        rx, dx = 0, 0
        if isfit == True:
            self.skip_th_range[myhash+'_'+str(ii[0])] = np.log2(np.max(dmse)) / 80
        try:
            aa = self.skip_th_range[myhash+'_'+str(ii[0])]
            e, s = 80, 1
            print("SKIP TH RANGE", e, s, np.power(2, aa*float(80)))
        except:
            e = self.skip_th_range.get(myhash+'_'+str(0)+'_'+str(ii[0]), 60000)
            s = self.skip_th_step.get(myhash+'_'+str(0)+'_'+str(ii[0]),100)
            print("SKIP TH RANGE", e, s)
        
        for k in range(0, (int)(e), (int)(s)):
            if s == 1:
                skip_TH = np.round(np.power(2, aa*float(k))*100)/100
            else:
                skip_TH = k
            
            idx = (dmse > skip_TH).reshape(-1)
            if np.sum(idx) == 0:
                if is0 == True:
                    break
                else:
                    is0 = True
            if isfit == True or self.fast==True:
                st0=''
                l = 2*2.75*min(1-np.sum(idx)/len(idx),np.sum(idx)/len(idx))+0.00276
                l = int(l*len(idx))
                for _ in range(l+1):
                    st0+='0'
            else:
                st0 = HierarchyCABAC().encode(None, idx.reshape(S), 1) 
            st1 = ''
            for i in range(len(ii)):
                if isfit== True:
                    km = self.myKMeans[myhash][ii[i]]
                    nc = km.n_clusters
                    self.Huffman[myhash+'_'+str(i)+'_'+str(ii[i])+'_'+str(skip_TH)+'_h'] = Huffman().fit(label.reshape(-1)[idx.reshape(-1)].tolist() + np.arange(nc).tolist())
                    self.Huffman[myhash+'_'+str(i)+'_'+str(ii[i])+'_'+str(skip_TH)] = VQEntropy(nc, km.inverse_predict(np.arange(nc).reshape(-1, 1))).fit(label.reshape(S), idx.reshape(S))
#                     continue
                h1 = self.Huffman.get(myhash+'_'+str(i)+'_'+str(ii[i])+'_'+str(skip_TH), None)
                h2 = self.Huffman.get(myhash+'_'+str(i)+'_'+str(ii[i])+'_'+str(skip_TH)+'_h', None)
                if h1 is not None:
                    st1 = h1.encode(label.reshape(S), idx.reshape(S))
                    if h2 is not None and isfit == False:
                        b = h2.encode(label.reshape(-1)[idx.reshape(-1)])
                        if len(st1) > len(b):
                            st1 = b
                else:
                    #print('skip')
                    # no exist, skip all the current index
                    # maybe we need to use fixed length coding?
                    idx = idx.astype('int16')
                    idx *= 0
                    st0 = ''
            r = len(st0+st1) 
            # compute the distortion by zero out the skipped ones
            d = np.zeros_like(mse)
            d[idx.reshape(-1)] += mse[idx.reshape(-1)]
            d[idx.reshape(-1)==False] += omse[idx.reshape(-1)==False]
            d = np.mean(d)
            # using r is actually the bit incremental, but the previous would not change
            # which is a constant, so that we can ignore it
            cost = d + self.Lagrange_multip * r/ S[0] /1024**2 * pow(1.3, 8-int(myhash[1]))
            if min_cost > cost:
                min_cost = cost
                th = skip_TH
                rx, dx = len(st0+st1), d
                sidx= idx
            if lcost <= cost:
                break
            else:
                lcost = cost
        return th, [min_cost, rx, dx], sidx

    # compute the rd cost for given iX
    def RD(self, tX, X, iX, label, level, pos, pidx=None, ii=[], isfit=False):
        myhash = 'L'+str(level)+'-P'+str(pos)
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        siX = np.zeros_like(X)
        siX += iX
        if self.fast == True: # actual encoding time, the global mse is better
            sX, siX = self.to_spatial(X, tX, level, True), self.to_spatial(siX, tX, level, False)
            acc_win = 1
            for i in range(0, level+1):
                acc_win *= self.win_list[i]
            sX, siX = Shrink(sX, acc_win), Shrink(siX, acc_win)
        else:
            sX, siX = self.to_spatial(X, tX, level, True), self.to_spatial(siX, tX, level, False)
            acc_win = 1
            for i in range(0, level+1):
                acc_win *= self.win_list[i]
            sX, siX = Shrink(sX, acc_win), Shrink(siX, acc_win)
        sX, siX = sX.reshape(-1, sX.shape[-1]), siX.reshape(-1, siX.shape[-1])
        if isMAD == True:
            mse = (np.mean(np.abs((sX-siX).astype('float32')),axis=1))
            omse =  np.mean(np.abs(sX.astype('float32')), axis=1)
        else:
            mse = (np.mean(np.square((sX-siX).astype('float32')),axis=1))
            omse =  np.mean(np.square(sX.astype('float32')), axis=1)
        dmse = omse-mse
        if self.isdistributed[0] > -1:
            write_pkl(self.isdistributed[1]+'/kmidx_'+str(self.isdistributed[2])+'/'+str(self.isdistributed[0])+'.dmse', dmse.reshape(S))
            write_pkl(self.isdistributed[1]+'/kmidx_'+str(self.isdistributed[2])+'/'+str(self.isdistributed[0])+'.label', label.reshape(S))
            self.max_dmse[self.isdistributed[2]] = max(np.max(dmse), self.max_dmse.get(self.isdistributed[2], -1))
        th, cost, idx = self.RD_search_th(myhash, dmse, mse, omse, pidx, label, S, ii, isfit)
        return th, cost, idx

    # for each content select suitable codebook
    @Time
    def RD_search_km(self, tX, X, level, pos, pidx, isfit):
        myhash = 'L'+str(level)+'-P'+str(pos)
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        X = X.reshape(-1, X.shape[-1])
        TH, min_cost, skip_idx, tiX = 0, [1e20], None, None
        lcost = 1e40
        for i0 in range(len(self.myKMeans[myhash])):
            km = self.myKMeans[myhash][i0]
            iX = np.zeros_like(X).reshape(-1, X.shape[-1])
            label = km.predict(X[:,:self.n_dim_list[level][pos]]).reshape(-1)
            iX[:,:self.n_dim_list[level][pos]] = km.inverse_predict(label.reshape(-1,1))
            th, cost, idx = self.RD(tX, X.reshape(S), iX.reshape(S), label, level, pos, pidx, [i0], isfit)
            if cost[0] < min_cost[0]:
                TH, min_cost, skip_idx = th, cost, idx
                tiX = iX
            if lcost < cost[0]:
                break
            else:
                lcost = cost[0]
        self.buffer['TH'] = TH
        self.buffer['i0'] = i0
        if self.isdistributed[0] > -1:
            write_pkl(self.isdistributed[1]+'/kmidx_'+str(self.isdistributed[2])+'/'+str(self.isdistributed[0])+'.idx', skip_idx.reshape(S))
        myLog('<INFO> RD_cost=%8.4f r=%f d=%4.5f Skip_TH=%f'%(min_cost[0], min_cost[1], min_cost[2], TH))
        tiX = tiX.reshape(-1, tiX.shape[-1])
        tiX[skip_idx ==  False] *= 0 
        myLog('<BITSTREAM> bpp=%f'%min_cost[1])
        self.acc_bpp += min_cost[1] 
        return tiX.reshape(S)
        
    @Time
    def fit_one_level_one_pos(self, X, tX, level, pos):
        myhash = 'L'+str(level)+'-P'+str(pos)
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], X.shape[-1])
        myLog('id=%s vq_dim=%d n_clusters=%d'%(myhash, self.n_dim_list[level][pos], self.n_clusters_list[level][pos]))
        S = X.shape
        iX = np.zeros_like(X).reshape(-1, X.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        for i in range(1):
            nc = self.n_clusters_list[level][pos]
            tmp, tmp_h = [], []
            while nc > 1:
                km = myKMeans(nc).fit(X[:,:self.n_dim_list[level][pos]])
                tmp.append(km)
                nc = nc //2
            self.myKMeans[myhash+'_'+str(i)] = tmp
        X, iX = X.reshape(S), iX.reshape(S)
        iX = self.RD_search_km(tX, X, level, pos, None, True)
        X[:, :,:,:self.n_dim_list[level][pos]] -= iX[:, :,:,:self.n_dim_list[level][pos]]
        return X
    

    def fit_vq_entropy_distributed(self, root, n_file, myhash):
        self.skip_th_range[myhash+'_'+str(self.isdistributed[2])] = np.log2(np.max(self.isdistributed[2])) / 80
        km = self.myKMeans[myhash][self.isdistributed[2]]
        nc = km.n_clusters
        self.Huffman[myhash+'_'+str(self.isdistributed[2])+'_h'] = Huffman().fit_distributed(root+'/kmidx_'+str(self.isdistributed[2]), n_file, nc)
        self.Huffman[myhash+'_'+str(self.isdistributed[2])] = VQEntropy(nc, km.inverse_predict(np.arange(nc).reshape(-1, 1))).fit_distributed(root+'/kmidx_'+str(self.isdistributed[2]), 
                                                                                                                                              n_file, 
                                                                                                                                              skrange=self.max_dmse[self.isdistributed[2]])
#                     continue

    def fit_one_level_one_pos_distributed(self, root, n_file, level, pos):
        myhash = 'L'+str(level)+'-P'+str(pos)
        # myLog('id=%s vq_dim=%d n_clusters=%d'%(myhash, self.n_dim_list[level][pos], self.n_clusters_list[level][pos]))
        dim = -1
        for fileID in range(n_file):
            X = load_pkl(root+'/'+str(fileID)+'.iR')
            write_pkl(root+'/'+str(fileID)+'.data', X.reshape(-1, X.shape[-1]))
            dim = X.shape[-1]
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], dim)
        nc = self.n_clusters_list[level][pos]
        dkm = mydKMeans(nc, self.n_dim_list[level][pos])
        for n_iter in range(100000):
            dkm.fit(root, n_file)
        X = []
        for fileID in range(min(5,n_file)):
            X.append(load_pkl(root+'/'+str(fileID)+'.data'))
        self.myKMeans[myhash] = [dkm.KM] + split_km_subspace(dkm.KM, np.concatenate(X,axis=0))
        os.system('rm -rf *.data')
        for kmidx in range(len(self.myKMeans[myhash])):
            os.system('mkdir '+root+'/kmidx_'+str(kmidx))
            for fileID in range(n_file):
                self.isdistributed = [fileID, root, kmidx]
                X = load_pkl(root+'/'+str(fileID)+'.iR')
                tX = load_pkl(root+'/'+str(fileID)+'.cwsaab')
                iX = self.RD_search_km(tX, X, level, pos, None, True)
                X[:,:,:,:self.n_dim_list[level][pos]] -= iX[:, :,:,:self.n_dim_list[level][pos]]
                write_pkl(root+'/'+str(fileID)+'.iR', X)
            self.fit_vq_entropy_distributed(root, n_file, myhash)

    @Time
    def fit_one_level(self, iR, tX, level):
        myhash = 'L'+str(level)
        self.shape[myhash] = [iR.shape[0], iR.shape[1], iR.shape[2], -1]
        myLog('id=%s'%myhash)
        for pos in range(len(self.n_dim_list[level])):
            iR = self.fit_one_level_one_pos(iR, tX, level, pos)
        return iR.reshape(self.shape[myhash])
    
    @Time
    def fit_one_level_distributed(self, root, n_file, level):
        for pos in range(len(self.n_dim_list[level])):
            self.fit_one_level_one_pos_distributed(root, n_file, level, pos)


    @Time
    def fit(self, X):
        self.isfit=True
        self.cwSaab.fit(X)
        tX = self.cwSaab.transform(X)
        iR = tX[-1]
        for level in range(len(self.n_dim_list)-1, -1, -1):
            iR = self.fit_one_level(iR, tX, level)
            if level > 0:
                iR = self.cwSaab.inverse_transform_one(iR, tX[level-1], level)
            else:
                iR = self.cwSaab.inverse_transform_one(iR, None, level)
        self.isfit=False
        return iR

    def fit_distributed(self, root, n_file):
        self.isfit=False
        X = []
        # cwsaab distributed fit not supported
        for fileID in range(min(5, n_file)):
            X.append(load_pkl(root+'/'+str(fileID)+'.spatial_data'))
        X = np.concatenate(X, axis=0)
        self.cwSaab.fit(X)
        self.cwSaab.transform_distributed(root, n_file)
        for level in range(len(self.n_dim_list)-1, -1, -1):
            self.fit_one_level_distributed(root, n_file, level)
            self.cwSaab.inverse_transform_one_distributed(root, n_file, level)
        self.isfit=False
    

    def predict_one_level_one_pos(self, tX, X, level, pos, gidx, skip):
        myhash = 'L'+str(level)+'-P'+str(pos)
        myLog('id=%s'%(myhash))
        if myhash in skip:
            myLog('<INFO> SKIP')
            return X
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], X.shape[-1])
        myLog('id=%s vq_dim=%d n_clusters=%d'%(myhash, self.n_dim_list[level][pos], self.n_clusters_list[level][pos]))
        S = X.shape
        X = X.reshape(-1, X.shape[-1])
        X = X.reshape(S)
        iX = self.RD_search_km(tX, X, gidx, level, pos, self.buffer.get('L'+str(level+1)+'-P'+str(0)+'_idx', None), False)
        X[:, :,:,:self.n_dim_list[level][pos]] -= iX[:, :,:,:self.n_dim_list[level][pos]]
        
        return X
    
    #@Time
    def predict_one_level(self, tX, iR, level, skip=[]):
        myhash = 'L'+str(level)
        self.shape[myhash] = [iR.shape[0], iR.shape[1], iR.shape[2], -1]
        gidx = self.myKMeans[myhash].predict(iR)
        for pos in range(len(self.n_dim_list[level])):
            iR = self.predict_one_level_one_pos(tX, iR, level, pos, gidx,skip)
        return iR.reshape(self.shape[myhash])

    def predict(self, X, skip=[],fast=True):
        print('here',self.Lagrange_multip)
        
        if X.shape[0]>300:
            self.fast=True
        else:
            self.fast=False
        self.buffer = {}
        self.S = []
        tX = self.cwSaab.transform(X)
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

    