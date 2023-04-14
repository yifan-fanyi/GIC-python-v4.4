from core.util.ac import HierarchyCABAC
from core.util.Huffman import Huffman
from core.util.myKMeans import myKMeans
from core.util.evaluate import MSE
from core.util import myLog, Shrink, invShrink
import numpy as np
import pickle
from core.util.ReSample import resize

class spatialKM():
    def __init__(self, grid, model_hash, win_list, n_clusters_list, skip_TH_list):
        self.grid = grid
        self.model_hash = model_hash
        self.win_list = win_list
        self.n_clusters_list = n_clusters_list
        self.skip_TH_list = skip_TH_list
        self.model = {}
        self.buffer = {}
        self.loaded = False
        self.root = './cache/'

    def select(self, X, iX, pos):
        S = list(X.shape)
        S[-1]=-1
        X = X.reshape(-1,X.shape[-1])
        iX = iX.reshape(-1, iX.shape[-1])
        dmse = (np.mean(np.square(X), axis=1)-np.mean(np.square(X-iX),axis=1))
        idx = dmse > self.skip_TH_list[pos]
        iX[idx.reshape(-1) == False] *= 0
        idx, iX = idx.reshape(S), iX.reshape(S)
        if X.shape[0] < 200:
            ac = HierarchyCABAC()
            myLog('<BITSTREAM> %s skip_idx=%d'%('P'+str(pos), len(ac.encode(self.buffer.get(str(pos-1)+'_idx',None), idx, 1))))
        return idx, iX

    def load(self):
        ct = len(self.win_list)
        try:
            with open(self.root+'G'+str(self.grid)+'add_'+self.model_hash+'.pkl', 'rb') as f:
                self.model = pickle.load(f)
                for i in range(len(self.win_list)):
                    if str(self.n_clusters_list[i])+'_'+str(self.win_list[i]) in self.model.keys():
                        ct -= 1
        except:
            self.model = {}
        if ct == 0:
            self.loaded = True

    def fit(self, iX, X):
        myLog('---> s Spatial KM fit')
        self.load()
        if self.loaded == True:
            return self.predict(iX, X)
        R = X - resize(iX, X.shape[1])
        for i in range(len(self.win_list)):
            R = Shrink(R, self.win_list[i])
            m_hash = str(self.n_clusters_list[i])+'_'+str(self.win_list[i])
            m = self.model.get(m_hash, {})
            if 'km' not in m.keys():
                m['km'] = myKMeans(self.n_clusters_list[i]).fit(R)
            label = m['km'].predict(R)
            iR = m['km'].inverse_predict(label)
            idx, iR = self.select(R, iR, i)
            if 'h' not in m.keys():
                m['h'] = Huffman().fit(label.reshape(-1)[idx.reshape(-1)])
            self.buffer[str(i)+'_idx'] = idx
            R -= iR
            R = invShrink(R, self.win_list[i])
            self.model[m_hash] = m
        with open(self.root+'G'+str(self.grid)+'add_'+self.model_hash+'.pkl', 'wb') as f:
            pickle.dump(self.model, f, 4)
        myLog('---> e Spatial KM fit')
        return X - R
    
    def predict(self, iX, X, refX=None):
        self.load()
        R = X - resize(iX, X.shape[1])
        for i in range(len(self.win_list)):
            m_hash = str(self.n_clusters_list[i])+'_'+str(self.win_list[i])
            m = self.model[m_hash]
            R = Shrink(R, self.win_list[i])
            label = m['km'].predict(R)
            iR = m['km'].inverse_predict(label)
            idx, iR = self.select(R, iR, i)
            self.buffer[str(i)+'_st'] = m['h'].encode(label.reshape(-1)[idx.reshape(-1)])
            self.buffer[str(i)+'_idx'] = idx
            myLog('<BITSTREAM> st=%d'%len(self.buffer[str(i)+'_st']))
            R -= iR
            R = invShrink(R, self.win_list[i])
        iX = X - R
        myLog('<INFO> local MSE=%f'%MSE(X, iX))
        if refX is not None:
            myLog('<MSE> global MSE=%f'%MSE(refX, resize(iX, refX.shape[1])))
        return iX

