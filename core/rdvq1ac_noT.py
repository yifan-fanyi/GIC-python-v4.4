from core.cwSaab import cwSaab
from core.util.myKMeans import myKMeans
from core.util.Huffman import Huffman
import numpy as np
from core.util import Time, myLog, Shrink, invShrink
from core.util.ac import HierarchyCABAC, BAC
from core.util.evaluate import MSE
from core.util.ReSample import *
from core.VQEntropy import VQEntropy
from core.Distributed_KMeans import Distributed_KMeans

isMAD = False
print('<FRAMEWORK> rdVQ1 2022.12.09', isMAD)

def toSpatial(cwSaab, iR, level, S,tX):
    for i in range(level, -1, -1):
        if i > 0:
            iR = cwSaab.inverse_transform_one(iR, tX[i-1], i)
        else:
            iR = cwSaab.inverse_transform_one(iR, None, i)
    return iR
def get_dmse(km, AC):    
    label = km.predict(AC)
    iAC = km.inverse_predict(label)
    sX, siX = AC.reshape(-1, AC.shape[-1]), iAC.reshape(-1, iAC.shape[-1])
    mse = (np.mean(np.square((sX-siX).astype('float32')),axis=1))
    omse =  np.mean(np.square(sX.astype('float32')), axis=1)
    dmse = omse-mse
    dmse = dmse.reshape(label.shape)
    return label, iAC, dmse
def split_km_subspace(KM, dmse, label, win):
    # label, _, dmse = get_dmse(KM, AC) 
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
    km_list = [myKMeans(-1).fit(X=None, cluster_centers=cent[:8]), 
                myKMeans(-1).fit(X=None, cluster_centers=cent[:16]), 
               myKMeans(-1).fit(X=None, cluster_centers=cent[:32]),
               myKMeans(-1).fit(X=None, cluster_centers=cent[:64]),
               myKMeans(-1).fit(X=None, cluster_centers=cent[:128]),
               myKMeans(-1).fit(X=None, cluster_centers=cent[:256]),
               myKMeans(-1).fit(X=None, cluster_centers=cent[:1024]),
               myKMeans(-1).fit(X=None, cluster_centers=cent[:2048]),
              KM]
    return km_list

class VQ_noT:
    def __init__(self, n_clusters_list, win_list, n_dim_list, Lagrange_multip=300000, acc_bpp=0):
        self.n_clusters_list = n_clusters_list
        self.win_list = win_list
        self.n_dim_list = n_dim_list
        self.shape = {}
        self.myKMeans = {}
        self.Huffman = {}
        self.buffer = {}
        self.acc_bpp = acc_bpp
        self.Lagrange_multip = Lagrange_multip
        self.fast=True
        self.skip_th_range = {}
        self.end = False
        self.th = -1

    def get_myhash(self, level, pos=-1, ispartial=False):
        if ispartial == True:
            return 'L'+str(level)
        return 'L'+str(level)+'-P'+str(pos)
    # find the optimal threshold
    def RD_search_th(self, myhash, dmse, mse, omse, pidx, label, S, kmidx, isfit):
        min_cost, th, lcost = 1e40, -1, 1e40
        is0 = False
        rx, dx = 0, 0
        if isfit == True:
            self.skip_th_range[myhash+'_'+str(kmidx)] = np.log2(np.max(dmse)*0.9) / 80
        aa = self.skip_th_range[myhash+'_'+str(kmidx)]
        e, s = 80, 1
        myLog("SKIP TH RANGE (%f, %f, %f)"%(np.power(2, aa*float(s)), np.power(2, aa*float(e)), aa))
        if isfit == True:
            for k in range(0, 3000,5):
                skip_TH = np.round(np.power(2, aa*float(k))*1000)/1000
   
                idx = (dmse > skip_TH).reshape(-1)
                km = self.myKMeans[myhash][kmidx]
                nc = km.n_clusters
                if k == 0:
                    self.Huffman[myhash+'_'+str(kmidx)] = \
                        VQEntropy(nc, km.inverse_predict(np.arange(nc).reshape(-1, 1))).fit(
                            label.reshape(S), idx.reshape(S), keep_fit=False, done=False)
                elif k > 2994: # last skip_th
                    self.Huffman[myhash+'_'+str(kmidx)].fit(
                            label.reshape(S), idx.reshape(S), keep_fit=True, done=True)
                else:
                    self.Huffman[myhash+'_'+str(kmidx)].fit(
                            label.reshape(S), idx.reshape(S), keep_fit=True, done=False)
                # return
        is0 = False         
        # encode
        for k in range(0, (int)(e), (int)(s)):
            if s == 1:
                skip_TH = np.round(np.power(2, aa*float(k))*1000)/1000
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
                for i in range(l+1):
                    st0+='0'
            else:
                st0 = HierarchyCABAC().encode(None, idx.reshape(S), 1) 
            st1 = self.Huffman[myhash+'_'+str(kmidx)].encode(label.reshape(S), idx.reshape(S))
            b = self.Huffman[myhash+'_'+str(kmidx)].Huffman.encode(label.reshape(-1)[idx.reshape(-1)])
            # myLog('condition coding %d, huffman %d'%(len(st1), len(b)))
            if len(st1) > len(b):
                st1 = b
            r = len(st0+st1) / S[0] 
            d = np.zeros_like(mse)
            d[idx.reshape(-1)] += mse[idx.reshape(-1)] 
            d[idx.reshape(-1)==False] += omse[idx.reshape(-1)==False]
            d = np.mean(d)
            cost = d + self.Lagrange_multip * r /1024**2 * pow(1.3, 8-int(myhash[1]))
            if min_cost > cost:
                min_cost, th, sidx = cost,skip_TH, idx
                rx, dx = len(st0+st1), d
            if lcost <= cost:
                if isfit==False:
                    break # early stop if not training 
            else:
                lcost = cost
        return th, [min_cost, rx, dx], sidx

    # compute the rd cost for given iX
    def RD(self, tX, X, iX, label, level, pos, pidx=None, kmidx=-1, isfit=False):
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        siX = np.zeros_like(X)
        siX += iX
        sX = X
        sX, siX = sX.reshape(-1, sX.shape[-1]), siX.reshape(-1, siX.shape[-1])
        mse = (np.mean(np.square((sX-siX).astype('float32')),axis=1))
        omse =  np.mean(np.square(sX.astype('float32')), axis=1)
        dmse = omse-mse
        th, cost, idx = self.RD_search_th(self.get_myhash(level, pos), dmse, mse, omse, pidx, label, S, kmidx, isfit)
        return th, cost, idx

    # for each content select suitable codebook
    @Time
    def RD_search_km(self, tX, X, level, pos, pidx, isfit):
        myhash = self.get_myhash(level, pos)
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        X = X.reshape(-1, X.shape[-1])
        TH, min_cost, skip_idx, tiX, km_idx, lcost = 0, [1e20], None, None, -1, 1e40
        for kmidx in range(len(self.myKMeans[myhash])):
            iX = np.zeros_like(X).reshape(-1, X.shape[-1])
            km = self.myKMeans[myhash][kmidx]
            label = km.predict(X[:,:self.n_dim_list[level][pos]]).reshape(-1)
            iX[:,:self.n_dim_list[level][pos]] = km.inverse_predict(label.reshape(-1,1))
            th, cost, idx = self.RD(tX, X.reshape(S), iX.reshape(S), label, level, pos, pidx, kmidx, isfit)
            print('--local-optimal--',cost, th)
            if cost[0] < min_cost[0]:
                TH, min_cost, skip_idx, tiX = th, cost, idx, iX
                km_idx = kmidx
            if lcost < cost[0]:
                if isfit == False:
                    break
            else:
                lcost = cost[0]
        self.th = TH
        myLog('<INFO> RD_cost=%8.4f r=%f d=%4.5f Skip_TH=%d'%(min_cost[0], min_cost[1], min_cost[2], TH))
        tiX = tiX.reshape(-1, tiX.shape[-1])
        tiX[skip_idx ==  False] *= 0 
        myLog('<BITSTREAM> bpp=%f'%min_cost[1])
        self.acc_bpp += min_cost[1] 
        return tiX.reshape(S)
    def get_split(self, km, X, level, pos):
        label = km.predict(X[:,:self.n_dim_list[level][pos]]).reshape(-1)
        ix = km.inverse_predict(label.reshape(-1,1))
        sX, siX = X.reshape(-1, X.shape[-1]), ix.reshape(-1, ix.shape[-1])
        mse = (np.mean(np.square((sX-siX).astype('float32')),axis=1))
        omse =  np.mean(np.square(sX.astype('float32')), axis=1)
        dmse = omse-mse
        return split_km_subspace(km, dmse, label, win=self.win_list[level])
    @Time
    def fit_one_level_one_pos(self, X, tX, level, pos):
        myhash = self.get_myhash(level, pos)
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], X.shape[-1])
        myLog('id=%s vq_dim=%d n_clusters=%d'%(myhash, self.n_dim_list[level][pos], self.n_clusters_list[level][pos]))
        S = X.shape
        X = X.reshape(-1, X.shape[-1])
        # nc = self.n_clusters_list[level][pos]
        # tmp = []
        # km = myKMeans(nc, KKZinit=True).fit(X[:,:self.n_dim_list[level][pos]])
        # dkm = Distributed_KMeans(self.n_clusters_list[0], size=X.shape[1], win=self.win_list[0], datatype='SHORT', frame_each_file=1000, n_frames=500, max_iter=10000, max_err=1e-7)
        # dkm.fit(self.FOLDER, self.FILE_LIST)
        # km = dkm.KM
        # tmp.append(km)
       
        self.myKMeans[myhash] += self.get_split(self.myKMeans[myhash][0], X, level, pos) 
        X = X.reshape(S)
        iX = self.RD_search_km(tX, X, level, pos, self.buffer.get('L'+str(level+1)+'-P'+str(0)+'_idx', None), True)
        X[:, :,:,:self.n_dim_list[level][pos]] -= iX[:, :,:,:self.n_dim_list[level][pos]]
        return X
    
    @Time
    def fit_one_level(self, iR, tX, level):
        myhash = self.get_myhash(level, ispartial=True)
        self.shape[myhash] = [iR.shape[0], iR.shape[1], iR.shape[2], -1]
        myLog('id=%s'%myhash)
        for pos in range(len(self.n_dim_list[level])):
            iR = self.fit_one_level_one_pos(iR, tX, level, pos)
        return iR.reshape(self.shape[myhash])

    @Time
    def fit(self, dkm, X):
        self.isfit=True
        self.myKMeans[self.get_myhash(0, 0)] = [dkm.KM]
        X = Shrink(X, self.win_list[0])
        iR = self.fit_one_level(X, None, 0)
        self.isfit=False
        return iR

    def predict_one_level_one_pos(self, tX, X, level, pos, skip):
        myhash = self.get_myhash(level, pos)
        myLog('id=%s'%(myhash))
        if myhash in skip:
            myLog('<INFO> SKIP CURRENT POS ->%s'%myhash)
            return X
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], X.shape[-1])
        myLog('id=%s vq_dim=%d n_clusters=%d'%(myhash, self.n_dim_list[level][pos], self.n_clusters_list[level][pos]))
        S = X.shape
        X = X.reshape(-1, X.shape[-1])
        X = X.reshape(S)
        iX = self.RD_search_km(tX, X, level, pos, self.buffer.get('L'+str(level+1)+'-P'+str(0)+'_idx', None), False)
        X[:, :,:,:self.n_dim_list[level][pos]] -= iX[:, :,:,:self.n_dim_list[level][pos]]
        return X
    
    #@Time
    def predict_one_level(self, tX, iR, level, skip=[]):
        myhash = self.get_myhash(level, ispartial=True)
        self.shape[myhash] = [iR.shape[0], iR.shape[1], iR.shape[2], -1]
        for pos in range(len(self.n_dim_list[level])):
            iR = self.predict_one_level_one_pos(tX, iR, level, pos,skip)
        return iR.reshape(self.shape[myhash])

    def predict(self, X, skip=[],fast=True):
        if X.shape[0]>300:
            self.fast=True
        else:
            self.fast=False
        self.buffer = {}
        X = Shrink(X, self.win_list[0])
        iR = self.predict_one_level([X], X, 0, skip)   
        iR = invShrink(X, self.win_list[0])
        return iR
