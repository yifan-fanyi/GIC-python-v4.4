import numpy as np
from core.util.myKMeans import myKMeans
from core.util.Huffman import Huffman
from core.util.ac import HierarchyCABAC
from core.util import myLog

class VQ:
    def __init__(self, n_clusters, dim, Lagrange_multip):
        self.n_clusters = n_clusters
        self.dim = dim
        self.vq_codebook = []
        self.vq_entropy = []
        self.skip_TH_low = 0
        self.skip_TH_high = 100000
        self.skip_TH_step = 200
        self.fast = True 
        self.BITSTREAM = {}
        self.Lagrange_multip = Lagrange_multip
    
    def fit(self, X):
        X = X.reshape(-1, X.shape[-1])
        nc = 1
        while nc <= self.n_clusters:
            km = myKMeans(nc).fit(X[:,:self.dim])
            self.vq_codebook.append(km)
            label = km.predict(X[:,:self.dim])
            self.vq_entropy.append(Huffman().fit(label))
        return self

    def RD_search(self, dmse, mse, omse, pidx, label, codebook_idx, S):
        min_cost, th, mr, md = 1e10, -1, 0, 0
        for skip_TH in range(self.skip_TH_low, self.skip_TH_high, self.skip_TH_step):
            idx = (dmse > skip_TH).reshape(-1)
            if self.fast == True:
                st0 = ''
                l = int((2.75*np.sum(idx)/len(idx)+0.00276*len(idx)))+1
                for i in range(l):
                    st0 += '0'
            else:
                st0 = HierarchyCABAC().encode(pidx, idx.reshape(S), 1) 
            st1 = self.vq_entropy[codebook_idx].encode(label.reshape(-1)[idx])
            r = len(st0+st1) / S[0] 
            d = np.zeros_like(mse)
            d[idx==True] = mse[idx==True]
            d[idx==False] = omse[idx==False]
            d = np.mean(d)
            cost = d + self.Lagrange_multip * r
            if min_cost > cost:
                min_cost = cost
                th, mr, md = skip_TH, r, d
        return th, [min_cost, mr, md]

    def skip_threshold_select(self, X, S, codebook_idx, pidx=None):
        label = self.vq_codebook[codebook_idx].predict(X[:, :self.dim])
        iX = np.zeros_like(X)
        iX[:, :self.dim] = self.vq_codebook[codebook_idx].inverse_predict(label)
        mse =  np.mean(np.abs(X-iX),axis=1)
        omse =  np.mean(np.abs(X), axis=1)
        dmse = omse - mse
        # find optional threshold for current codebook
        TH, cost = self.RD_search(dmse, mse, omse, pidx, label, codebook_idx, S)
        return TH, cost

    def codebook_select(self, X, S, pidx=None):
        min_cost = 1e10
        cache = []
        for codebook_idx in range(len(self.vq_codebook)):
            TH, cost = self.skip_threshold_select(X, S, codebook_idx, pidx)
            if cost[0] < min_cost:
                min_cost, cache = cost[0], [TH, cost]
        return codebook_idx, cache[0] # codebook_idx, skip_TH

    def actual_encode(self, X, skip_TH, codebook_idx, S, pidx=None):
        label = self.vq_codebook[codebook_idx].predict(X[:, self.dim])
        iX = np.zeros_like(X)
        iX[:, :self.dim] = self.vq_codebook[codebook_idx].inverse_predict(label)
        mse =  np.mean(np.abs(X-iX),axis=1)
        omse =  np.mean(np.abs(X), axis=1)
        dmse = omse - mse
        idx = (dmse > skip_TH).reshape(-1)
        label = label.reshape(-1)
        st0 = HierarchyCABAC().encode(pidx, idx.reshape(S), 1)
        st1 = self.vq_entropy[codebook_idx].encode(label[idx])
        self.BITSTREAM = {0:st0, 1:st1}
        iX[idx==False] *= 0 # set as 0
        myLog('<BITSTREAM> st_idx=%d'%len(st0))
        myLog('<BITSTREAM> st=%d'%len(st1))
        return iX

    def predict(self, X, pidx=None, fast=False, acc_bpp=0):
        self.fast, self.acc_bpp = fast, acc_bpp
        S = [X.shape[0], X.shape[1], X.shape[2], -1]
        # RD
        codebook_idx, skip_TH = self.codebook_select(X, S, pidx)
        # actual encode
        iX = self.actual_encode(X, skip_TH, codebook_idx, S, pidx)
        return iX.reshape(S)

    def getBITSTREAM(self):
        return self.BITSTREAM
    def getBITSTREAM_LEN(self):
        return len(self.BITSTREAM[0]) + len(self.BITSTREAM[1])
    def reduce(self):
        self.BITSTREAM = {}
        return self