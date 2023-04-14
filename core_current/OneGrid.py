import numpy as np
from core.LDPVQ import LDPVQ
from core.PQR import PQR
from core.util.ReSample import resize

class Forward:
    def __init__(self, size_list, color_transform_predict=True, parC=None):
        self.size_list = size_list
        self.color_transform_predict = color_transform_predict
        self.parC = parC
        self.PQR = PQR(toCluster=False,#parC['toCluster'], 
                        n_cent=parC['n_cent'])
    def fit(self, X):
        if self.color_transform_predict == True:
            self.PQR.fit(X, 2)
        return self

    def transform(self, X):
        DC, AC = [X], []
        if self.color_transform_predict == True:
            for i in range(1, len(self.size_list)):
                win = self.size_list[i] // self.size_list[i-1]
                mX, tX = self.PQR.transform(DC[-1], win)
                DC.append(mX)
                AC.append(tX)
        else:
            for i in range(1, len(self.size_list)):
                mX = resize(DC[-1], self.size_list[i])
                rX = resize(mX, self.size_list[i-1])
                AC.append(DC[-1]-rX)
                DC.append(mX)
        return DC, AC

    def inverse_transform(self, DC, AC):
        DC = DC[-1]
        if self.color_transform_predict == True:
            for i in range(len(self.size_list)-1, 0, 1):
                win = self.size_list[i-1] // self.size_list[i]
                DC = self.PQR.inverse_transform(DC, AC[i-1], win)
        else:
            for i in range(len(self.size_list)-1, 0, 1):
                DC = resize(DC, self.size_list[i-1]) + AC[i-1]
        return DC



class OneGrid:
    def __init__(self, parP, parQ, parR):
        self.pqr = myPCA(-1)
        self.parP = parP
        self.parQ = parQ
        self.parR = parR
        self.ldpvq_p = LDPVQ(win_list=parP['win_list'], 
                             par_n_cluster=parP['par_n_cluster'], 
                             par_n_dim=parP['par_n_dim'], 
                             Lagrange_multip=parP['Lagrange_multip'], 
                             transform_split=parP['transform_split'])
        self.ldpvq_q = LDPVQ(win_list=parQ['win_list'], 
                             par_n_cluster=parQ['par_n_cluster'], 
                             par_n_dim=parQ['par_n_dim'], 
                             Lagrange_multip=parQ['Lagrange_multip'], 
                             transform_split=parQ['transform_split'])
        self.ldpvq_r = LDPVQ(win_list=parR['win_list'], 
                             par_n_cluster=parR['par_n_cluster'], 
                             par_n_dim=parR['par_n_dim'], 
                             Lagrange_multip=parR['Lagrange_multip'], 
                             transform_split=parR['transform_split'])
        self.BITSTREAM = {}
        self.acc_bpp = 0

    def fit(self, X):
        self.pqr.fit(X.reshape(-1, X.shape[-1]))
        X = self.pqr.transform(X)
        self.ldpvq_p.fit(X[:,:,:,:1])
        self.ldpvq_q.fit(X[:,:,:,1:2])
        self.ldpvq_r.fit(X[:,:,:,2:])
        return self
    
    def predict(self, X, acc_bpp=0):
        X = self.pqr.transform(X)
        iRp = self.ldpvq_p.predict(X[:,:,:,:1], acc_bpp)
        self.acc_bpp = self.ldpvq_p.getACC_BPP()
        self.BITSTREAM['P'] = self.ldpvq_p.getBITSTREAM()
        iRq = self.ldpvq_q.predict(X[:,:,:,1:2], self.acc_bpp)
        self.acc_bpp = self.ldpvq_q.getACC_BPP()
        self.BITSTREAM['Q'] = self.ldpvq_q.getBITSTREAM()
        iRr = self.ldpvq_r.predict(X[:,:,:,1:2], self.acc_bpp)
        self.acc_bpp = self.ldpvq_r.getACC_BPP()
        self.BITSTREAM['R'] = self.ldpvq_r.getBITSTREAM()
        # return decoded image not residual
        return self.pqr.inverse_transform(X - np.concatenate([iRp, iRq, iRr],axis=-1))

    def getBITSTREAM(self):
        return self.BITSTREAM
    def getACC_BPP(self):
        return self.acc_bpp
    def reduce(self):
        self.ldpvq_p.reduce()
        self.ldpvq_q.reduce()
        self.ldpvq_r.reduce()
        self.BITSTREAM = {}
        self.acc_bpp = 0
        return self
        

