import numpy as np
from core.cwSaab import cwSaab
from core.DPVQ import DPVQ
class LDPVQ:
    def __init__(self, win_list, par_n_cluster, par_n_dim, Lagrange_multip, transform_split):
        self.par_n_cluster = par_n_cluster
        self.par_n_dim = par_n_dim
        self.Lagrange_multip = Lagrange_multip
        self.cwSaab = cwSaab(win=win_list, channelwise=True, transform_split=transform_split)
        self.dpvq_list = []
        self.BITSTREAM = {}
        self.acc_bpp = 0

    def fit(self, X):
        self.cwSaab.fit(X)
        tX = self.cwSaab.transform(X)
        iR = tX[-1]
        for level in range(len(self.n_dim_list)-1, -1, -1):
            dpvq = DPVQ(n_clusters_list=self.par_n_cluster[level], 
                        dim_list=self.par_n_dim[level], 
                        Lagrange_multip=self.Lagrange_multip).fit(iR)
            iR = dpvq.predict(iR)
            self.dpvq_list.append(dpvq)
            if level > 0:
                iR = self.cwSaab.inverse_transform_one(iR, tX[level-1], level)
            else:
                iR = self.cwSaab.inverse_transform_one(iR, None, level)            
        return iR

    def predict(self, X, acc_bpp=0):
        self.acc_bpp = acc_bpp
        tX = self.cwSaab.transform(X)
        iR = tX[-1]
        for level in range(len(self.n_dim_list)-1, -1, -1):
            iR = self.dpvq_list[level].predict(iR, self.acc_bpp)
            self.acc_bpp = self.dpvq_list[level].getACC_BPP()
            self.BITSTREAM[level] = self.dpvq_list[level].getBITSTREAM()
            if level > 0:
                iR = self.cwSaab.inverse_transform_one(iR, tX[level-1], level)
            else:
                iR = self.cwSaab.inverse_transform_one(iR, None, level)            
        return iR

    def getBITSTREAM(self):
        return self.BITSTREAM
    def getACC_BPP(self):
        return self.acc_bpp
    def reduce(self):
        for i in range(len(self.dpvq_list)):
            self.dpvq_list[i].reduce()
        self.BITSTREAM = {}
        self.acc_bpp = 0
        return self