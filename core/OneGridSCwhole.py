from core.util import Time, myLog
import pickle
import numpy as np
from core.util.myPCA import myPCA
from core.util.ReSample import resize
from core.util.evaluate import MSE

class OneGridSCwhole:
    def __init__(self, grid, model_hash, model_p=None, model_q=None, model_r=None):
        self.grid = grid
        self.model_hash = model_hash
        self.model_p = model_p
        self.loaded = False
        with open('name.pkl', 'rb') as f:
            d = pickle.load(f)
        self.root = d['root']

    def load(self):
        try:
            with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_1.model', 'rb') as f:
                d = pickle.load(f)
                self.model_p = d['P']
                self.loaded = True
        except:
            try:
                with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_1.model', 'rb') as f:
                    self.model_p = pickle.load(f)['P']
                self.model_p = self.model_p
            except:
                print('cannot load')
        try:
            self.model_p = self.clear(self.model_p)
        except:
            print('cannot clear')
    def clear(self, model):
        for k in model.Huffman.keys():
            for kk in model.Huffman[k].keys():
                if model.Huffman[k][kk] is not None:
                    model.Huffman[k][kk].clear()
                    model.Huffman[k][kk].cent = []
        return model
    
    @Time
    def fit(self, iX, rX, color=None):
        myLog('---------------s-----------------')
        myLog('Grid=%d'%self.grid)
        print(rX.shape)
        self.load()
        if self.loaded == True:
            return self
        iX = resize(iX, pow(2, self.grid))
        X = rX - iX
        try:
            with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_1.model', 'rb') as f:
                self.model_p = pickle.load(f)['P']
            self.model_p = None
        except:
            myLog("---> whole fit")
            self.model_p.fit(X)
            try:
                with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_1.model', 'wb') as f:
                    pickle.dump({'P':self.model_p},f,4)
                self.model_p = None
            except:
                pass
        myLog('---------------e-----------------')
        return 0#iX

    @Time
    def predict(self, iX, rX, color=None, refX=None, new_lambda=None):
        myLog('---------------s-----------------')
        myLog('Grid=%d'%self.grid)
        self.load()
        iX = resize(iX, pow(2, self.grid))
        X = rX - iX
        fast = False
        if iX.shape[0] > 200:
            fast = True
        if new_lambda is not None:
            self.model_p.Lagrange_multip = new_lambda[0]
        iR = self.model_p.predict(X, fast=fast)
        self.model_p.buffer  = {}
        iX = resize(iX,pow(2, self.grid)) + X - iR
        myLog('<INFO> local MSE=%f'%MSE(rX, iX))
        if refX is not None:
            myLog('<MSE> global MSE=%f'%MSE(refX, resize(iX, refX.shape[1])))
        myLog('---------------e-----------------')
        return iX


                                                    





