import numpy as np
from core.util.myPCA import myPCA
from core.util.myKMeans import myKMeans
from core.util import Shrink, invShrink
import matplotlib.pyplot as plt

class Evl_color():
    def energy_ratio(tX):
        ratio = []
        s  = []
        for i in range(len(tX)):
            t = tX[i].reshape(-1, tX.shape[-1]).astype('float64')
            t = np.mean(np.square(t), axis=0)
            s.append(np.sum(t))
            t = t/np.sum(t)
            ratio.append(t)
        ratio = np.array(ratio)
        plt.hist(s)
        plt.show()
        for i in range(ratio.shape[1]):
            plt.hist(ratio[:,i], bins=32)
            plt.title('Channel#'+str(i))
            plt.show()
        return ratio, s
    def eng(X):
        a = X.reshape(-1,X.shape[-1])
        return sorted(np.mean(np.square(a),axis=0), key=lambda x:-x)     
class PQR:
    def __init__(self, toCluster=False, n_cent=1):
        self.win = 32
        self.myPCA_list = []
        self.colorM = myKMeans(n_cent)
        self.toCluster = toCluster
        self.n_cent = n_cent

    def rep(self, blabel):
        for i in range(int(np.log2(self.win**2))):
            blabel = np.concatenate([blabel, blabel], axis=-1)
        blabel = invShrink(blabel, self.win)
        return blabel
    
    def get_mean(self, X):
        a = Shrink(X[:,:,:,:1], self.win)
        S = a.shape
        a = a.reshape(-1, a.shape[-1])
        m1 = np.mean(a, axis=-1, keepdims=True)  
        a = Shrink(X[:,:,:,1:2], self.win)
        a = a.reshape(-1, a.shape[-1])
        m2 = np.mean(a, axis=-1, keepdims=True) 
        a = Shrink(X[:,:,:,2:], self.win)
        a = a.reshape(-1, a.shape[-1])
        m3 = np.mean(a, axis=-1, keepdims=True) 
        return np.concatenate([m1,m2,m3],axis=1).reshape(S[0],S[1],S[2],-1)
    def remove_mean(self, X, m):
        a = Shrink(X[:,:,:,:1], self.win)
        a = invShrink(a-m[:,:,:,:1], self.win)
        b = Shrink(X[:,:,:,1:2], self.win)
        b = invShrink(b-m[:,:,:,1:2], self.win)
        c = Shrink(X[:,:,:,2:], self.win)
        c = invShrink(c-m[:,:,:,2:], self.win)
        return np.concatenate([a,b,c],axis=-1)
    def add_mean(self, X, m):
        a = Shrink(X[:,:,:,:1], self.win)
        a = invShrink(a+m[:,:,:,:1], self.win)
        b = Shrink(X[:,:,:,1:2], self.win)
        b = invShrink(b+m[:,:,:,1:2], self.win)
        c = Shrink(X[:,:,:,2:], self.win)
        c = invShrink(c+m[:,:,:,2:], self.win)
        return np.concatenate([a,b,c],axis=-1)
    
    def fit(self, tX, win=32):
        self.win = win
        X = tX.copy()
        mX = self.get_mean(X)
        self.mX = mX
        self.colorM.fit(mX)
        label = self.colorM.predict(mX)
        blabel = label.reshape(X.shape[0],X.shape[1]//self.win,X.shape[2]//self.win,1)
        blabel = self.rep(blabel)
        imX = self.colorM.inverse_predict(label)   
        if self.toCluster == True:
            X = self.remove_mean(X, imX)
            print('PQR ',self.win,np.mean(np.square(mX-imX)))   
        else:
            X = self.remove_mean(X, mX)
        X = X.reshape(-1, X.shape[-1])
        for i in range(self.n_cent):
            idx = blabel.reshape(-1) == i
            self.myPCA_list.append(myPCA(-1).fit(X[idx]))
        return self

    def transform2(self, tX):
        X = tX.copy()
        print('eng init',Evl_color.eng(X))   
        mX = self.get_mean(X)
        label = self.colorM.predict(mX)
        blabel = label.reshape(X.shape[0],X.shape[1]//self.win,X.shape[2]//self.win,1)
        blabel = self.rep(blabel)
        imX = self.colorM.inverse_predict(label) 
        print('PQR ',self.win,np.mean(np.square(mX-imX)))   
        X = self.remove_mean(X, imX)
        print('eng remove mean',Evl_color.eng(X)) 
        S = X.shape
        X = X.reshape(-1, X.shape[-1])
        tX = np.zeros_like(X)
        for i in range(self.n_cent):
            idx = blabel.reshape(-1) == i
            if np.sum(idx) < 1:
                continue
            tX[idx] = self.myPCA_list[i].transform(X[idx])
        print('eng transform',Evl_color.eng(tX))
        return label, tX.reshape(S)
    
    def transform1(self, tX):
        X = tX.copy()
        print('eng init',Evl_color.eng(X))   
        mX = self.get_mean(X)
        label = self.colorM.predict(mX)
        blabel = label.reshape(X.shape[0],X.shape[1]//self.win,X.shape[2]//self.win,1)
        blabel = self.rep(blabel)
        imX = self.colorM.inverse_predict(label) 
        X = self.remove_mean(X, mX)
        print('eng remove mean',Evl_color.eng(X)) 
        S = X.shape
        X = X.reshape(-1, X.shape[-1])
        tX = np.zeros_like(X)
        for i in range(self.n_cent):
            idx = blabel.reshape(-1) == i
            if np.sum(idx) < 1:
                continue
            tX[idx] = self.myPCA_list[i].transform(X[idx])
        print('eng transform',Evl_color.eng(tX))
        return mX, tX.reshape(S)
    
    def inverse_transform2(self, label, tX):
        S = tX.shape
        tX = tX.reshape(-1, tX.shape[-1])
        X = np.zeros_like(tX)
        blabel = label.reshape(S[0],S[1]//self.win,S[2]//self.win,1)
        blabel = self.rep(label)
        for i in range(self.n_cent):
            idx = blabel.reshape(-1) == i
            if np.sum(idx) < 1:
                continue
            X[idx] = self.myPCA_list[i].inverse_transform(tX[idx])
        imX = self.colorM.inverse_predict(label)   
        X = self.add_mean(X.reshape(S), imX)
        return X
    
    def inverse_transform1(self, mx, tX):
        S = tX.shape
        tX = tX.reshape(-1, tX.shape[-1])
        X = np.zeros_like(tX)
        label = self.colorM.predict(mx)
        blabel = label.reshape(S[0],S[1]//self.win,S[2]//self.win,1)
        blabel = self.rep(label)
        for i in range(self.n_cent):
            idx = blabel.reshape(-1) == i
            if np.sum(idx) < 1:
                continue
            X[idx] = self.myPCA_list[i].inverse_transform(tX[idx])
        X = self.add_mean(X.reshape(S), mx)
        return X

    def transform(self, tX, win=32):
        self.win = win
        if self.toCluster == True:
            return self.transform2(tX)
        else:
            return self.transform1(tX)
    def inverse_transform(self, mx, tX, win=32):
        self.win = win
        if self.toCluster == True:
            return self.inverse_transform2(mx,tX)
        else:
            return self.inverse_transform1(mx, tX)