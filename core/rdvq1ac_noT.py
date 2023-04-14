from core.cwSaab import cwSaab
from core.util.myKMeans import myKMeans
from core.util.Huffman import Huffman
import numpy as np
from core.util import Time, myLog, Shrink, invShrink
from core.util.ac import HierarchyCABAC, BAC
from core.util.evaluate import MSE
from core.util.ReSample import *
from core.VQEntropy import VQEntropy

isMAD = False
print('<FRAMEWORK> rdVQ1 2022.12.09', isMAD)

def toSpatial(cwSaab, iR, level, S,tX):
    for i in range(level, -1, -1):
        if i > 0:
            iR = cwSaab.inverse_transform_one(iR, tX[i-1], i)
        else:
            iR = cwSaab.inverse_transform_one(iR, None, i)
    return iR
def split_km_subspace(KM, n_subspace=3):
    eng = np.mean(np.square(KM.cluster_centers_),axis=1).reshape(-1,1)
    km = myKMeans(n_subspace).fit(eng)
    label = km.predict(eng).reshape(-1)
    km_list = [myKMeans(-1).fit(X=None, cluster_centers=KM.cluster_centers_[:64]), 
               myKMeans(-1).fit(X=None, cluster_centers=KM.cluster_centers_[:128]),
               myKMeans(-1).fit(X=None, cluster_centers=KM.cluster_centers_[:256]),
               myKMeans(-1).fit(X=None, cluster_centers=KM.cluster_centers_[:512]),
              myKMeans(-1).fit(X=None, cluster_centers=KM.cluster_centers_[64:]), 
               myKMeans(-1).fit(X=None, cluster_centers=KM.cluster_centers_[128:]),
               myKMeans(-1).fit(X=None, cluster_centers=KM.cluster_centers_[256:]),
               myKMeans(-1).fit(X=None, cluster_centers=KM.cluster_centers_[512:])
              ]
#     km_list = []
    for i in range(n_subspace):
        idx = label == i
        k = myKMeans(-1).fit(X=None, cluster_centers=KM.cluster_centers_[idx])
        print('nc',k.n_clusters)
        km_list.append(k)
    return km_list
class VQ_noT:
    def __init__(self, n_clusters_list, win_list, n_dim_list, enable_skip={}, transform_split=0,Lagrange_multip=300000, acc_bpp=0):
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
        self.skip_th_step = {}

    # find the optimal threshold
    def RD_search_th(self, myhash, dmse, mse, omse, pidx, label, S, gidx, h0, ii, isfit):
        min_cost, th = 1e40, -1
        is0 = False
        lcost=1e40
        rx, dx = 0, 0
        if isfit == True:
#             a = np.max(dmse)
#             b = a / 200
#             self.skip_th_range[myhash+'_'+str(0)+'_'+str(ii[0])] = (int)(a)
#             self.skip_th_step[myhash+'_'+str(0)+'_'+str(ii[0])] = (int)(b)
            self.skip_th_range[myhash+'_'+str(ii[0])] = np.log2(np.max(dmse)*0.7) / 80
    
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
                skip_TH = np.round(np.power(2, aa*float(k))*100000)/100000
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
            # no group index in this version
            # st2 = BAC().encode(gidx.reshape(-1)[idx.reshape(-1)==True]).output#h0.encode(gidx.reshape(-1)[idx.reshape(-1)==True])
            st1 = ''
            for i in range(len(ii)):
                if isfit== True:
                    km = self.myKMeans[myhash+'_'+str(0)][ii[i]]
                    nc = km.n_clusters
                    self.Huffman[myhash+'_'+str(i)+'_'+str(ii[i])+'_'+str(skip_TH)+'_h'] = Huffman().fit(label.reshape(-1)[idx.reshape(-1)].tolist() + np.arange(nc).tolist())
                    self.Huffman[myhash+'_'+str(i)+'_'+str(ii[i])+'_'+str(skip_TH)] = VQEntropy(nc, km.inverse_predict(np.arange(nc).reshape(-1, 1))).fit(label.reshape(S), idx.reshape(S))
#                     continue
            
                h1 = self.Huffman.get(myhash+'_'+str(i)+'_'+str(ii[i])+'_'+str(skip_TH), None)
                h2 = self.Huffman.get(myhash+'_'+str(i)+'_'+str(ii[i])+'_'+str(skip_TH)+'_h', None)
                if h1 is not None:
                    st1 = h1.encode(label.reshape(S), idx.reshape(S))
                    if h2 is not None:
                        b = h2.encode(label.reshape(-1)[idx.reshape(-1)])
                        if len(st1) > len(b):
                            st1 = b
                    
                else:
                    print('xxxx skip')
                    # no exist, skip all the current index
                    # maybe we need to use fixed length coding?
                    idx = idx.astype('int16')
                    idx *= 0
#             if isfit ==True:
#                 return 0,[0,0,0]
            r = len(st0+st1) / S[0] /1024**2
            # compute the distortion by zero out the skipped ones
            d = np.zeros_like(mse)
            d[idx.reshape(-1)] += mse[idx.reshape(-1)]
            d[idx.reshape(-1)==False] += omse[idx.reshape(-1)==False]
            d = np.mean(d)
            cost = d + self.Lagrange_multip * r * pow(1.3, 8-int(myhash[1]))
            print(skip_TH, r, d, cost)
            if min_cost > cost:
                min_cost = cost
                th = skip_TH
                sidx = idx
                rx, dx = len(st0+st1), d
            if lcost <= cost:
                break
            else:
                lcost = cost
        return th, [min_cost, rx, dx], sidx

    # compute the rd cost for given iX
    def RD(self, tX, X, iX, label, gidx, level, pos, pidx=None, ii=[], isfit=False):
        myhash = 'L'+str(level)+'-P'+str(pos)
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        siX = np.zeros_like(X)
        siX += iX
        sX = X
        sX, siX = sX.reshape(-1, sX.shape[-1]), siX.reshape(-1, siX.shape[-1])
        if isMAD == True:
            mse = (np.mean(np.abs((sX-siX).astype('float32')),axis=1))
            omse =  np.mean(np.abs(sX.astype('float32')), axis=1)
        else:
            mse = (np.mean(np.square((sX-siX).astype('float32')),axis=1))
            omse =  np.mean(np.square(sX.astype('float32')), axis=1)
        dmse = omse-mse
        th, cost, idx = self.RD_search_th(myhash, dmse, mse, omse, pidx, label, S, gidx, self.Huffman[myhash+'_gidx'], ii, isfit)
#         idx = (dmse > th).reshape(-1)
        return th, cost, idx

    # for each content select suitable codebook
    @Time
    def RD_search_km(self, tX, X, gidx, level, pos, pidx, isfit):
        myhash = 'L'+str(level)+'-P'+str(pos)
        label = np.zeros_like(gidx).reshape(-1)
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        X = X.reshape(-1, X.shape[-1])
        TH, min_cost, skip_idx, tiX = 0, [1e20], None, None
        lcost = 1e40
        for i0 in range(len(self.myKMeans[myhash+'_0'])):
            km = [self.myKMeans[myhash+'_0'][i0]]
            iX = np.zeros_like(X).reshape(-1, X.shape[-1])
            for i in range(len(km)):
                label[gidx.reshape(-1)==i] = km[i].predict(X[gidx.reshape(-1)==i,:self.n_dim_list[level][pos]]).reshape(-1)
                iX[gidx.reshape(-1)==i,:self.n_dim_list[level][pos]] = km[i].inverse_predict(label[gidx.reshape(-1)==i].reshape(-1,1))
            th, cost, idx = self.RD(tX, X.reshape(S), iX.reshape(S), label, gidx, level, pos, pidx, [i0], isfit)
            print('----',cost, th)
            if cost[0] < min_cost[0]:
                TH = th
                min_cost = cost
                skip_idx = idx
                tiX = iX
            if lcost < cost[0]:
                pass#break
            else:
                lcost = cost[0]
        myLog('<INFO> RD_cost=%8.4f r=%f d=%4.5f Skip_TH=%d'%(min_cost[0], min_cost[1], min_cost[2], TH))
        tiX = tiX.reshape(-1, tiX.shape[-1])
        tiX[skip_idx ==  False] *= 0 
        myLog('<BITSTREAM> bpp=%f'%min_cost[1])
        self.acc_bpp += min_cost[1] 
#         self.buffer[myhash+'_idx'] = skip_idx
#         self.buffer[myhash+'_label'] = label
#         self.buffer[myhash+'_gidx'] = gidx
#         self.buffer[myhash+'_th'] = TH
        return tiX.reshape(S)
        
    @Time
    def fit_one_level_one_pos(self, X, tX, level, pos, gidx):
        myhash = 'L'+str(level)+'-P'+str(pos)
        self.n_dim_list[level][pos] = min(self.n_dim_list[level][pos], X.shape[-1])
        myLog('id=%s vq_dim=%d n_clusters=%d'%(myhash, self.n_dim_list[level][pos], self.n_clusters_list[level][pos]))
        S = X.shape
        iX = np.zeros_like(X).reshape(-1, X.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        for i in range(1):
            ii = gidx.reshape(-1) == i
            nc = self.n_clusters_list[level][pos]
            tmp, tmp_h = [], []
            while nc > 1:
                km = myKMeans(nc).fit(X[ii,:self.n_dim_list[level][pos]])
                # label = km.predict(X[ii,:self.n_dim_list[level][pos]])
                # tmp_h.append(Huffman().fit(label))
                tmp.append(km)
                nc = nc //2
            self.myKMeans[myhash+'_'+str(i)] = tmp + split_km_subspace(tmp[0], n_subspace=3)
            # self.Huffman[myhash+'_'+str(i)] = tmp_h
        self.Huffman[myhash+'_gidx'] = Huffman().fit(gidx)
        X, iX = X.reshape(S), iX.reshape(S)
        iX = self.RD_search_km(tX, X, gidx, level, pos, self.buffer.get('L'+str(level+1)+'-P'+str(0)+'_idx', None), True)
        X[:, :,:,:self.n_dim_list[level][pos]] -= iX[:, :,:,:self.n_dim_list[level][pos]]
        return X
    
    @Time
    def fit_one_level(self, iR, tX, level):
        myhash = 'L'+str(level)
        self.shape[myhash] = [iR.shape[0], iR.shape[1], iR.shape[2], -1]
        myLog('id=%s'%myhash)
        self.myKMeans[myhash] = myKMeans(1).fit(iR)
        gidx = self.myKMeans[myhash].predict(iR)
        for pos in range(len(self.n_dim_list[level])):
            iR = self.fit_one_level_one_pos(iR, tX, level, pos, gidx)
        return iR.reshape(self.shape[myhash])

    @Time
    def fit(self, X):
        self.isfit=True
        X = Shrink(X, self.win_list[0])
        iR = self.fit_one_level(X, None, 0)
        self.isfit=False
        return iR

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
        gidx = np.zeros((iR.shape[0], iR.shape[1], iR.shape[2], 1))  #self.myKMeans[myhash].predict(iR)
        for pos in range(len(self.n_dim_list[level])):
            iR = self.predict_one_level_one_pos(tX, iR, level, pos, gidx,skip)
        return iR.reshape(self.shape[myhash])

    def predict(self, X, skip=[],fast=True):
        print('here',1.3)
        if X.shape[0]>300:
            self.fast=True
        else:
            self.fast=False
        self.buffer = {}
        X = Shrink(X, self.win_list[0])
        iR = self.predict_one_level([X], X, 0, skip)   
        iR = invShrink(X, self.win_list[0])
        return iR

    