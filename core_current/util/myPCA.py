# 2021.01.27
# @yifan
# PCA transformation 
#
# 2D PCA modified from https://blog.csdn.net/w450468524/article/details/54895477
#
import numpy as np
import faiss
from sklearn.decomposition import PCA

class myPCA():
    def __init__(self, n_components=-1, n_threads=8, whichPCA='sklearn', toint=False):
        self.n_components = n_components
        self.Kernels      = []
        self.PCA          = None
        self.Energy_ratio = []
        self.Energy       = []
        self.n_threads    = n_threads
        self.whichPCA     = whichPCA
        faiss.omp_set_num_threads(n_threads)
        self.toint = toint
        self.mult = 1
        self.__version__  = '2021.11.05' 

    def PCA_sklearn(self, X):
        self.PCA          = PCA(  n_components=self.n_components  )
        self.PCA.fit(X)
        self.Kernels      = self.PCA.components_
        #self.Energy_ratio = self.PCA.explained_variance_ratio_
        self.Energy       = self.PCA.explained_variance_
        if self.toint == True:
            self.mult = (int)(127 / np.max(np.abs(self.Kernels)))-1
            #print(self.mult)
            #print(self.Kernels)
            #print(np.round(self.Kernels*self.mult))
            self.Kernels =  np.round(self.Kernels*self.mult)/self.mult
           # print(self.Kernels)
        
    def PCA_faiss(self, X):
        X = X.astype('float32')
        self.PCA = faiss.PCAMatrix (X.shape[-1], self.n_components)
        self.PCA.train(X)
        tr = self.PCA.apply_py(X)
        self.Energy = (tr ** 2).sum(0)
        self.Energy_ratio = self.Energy / np.sum(self.Energy)

    def fit(self, X):
        X = X.reshape(  -1, X.shape[-1]  )
        if self.n_components < 0 or self.n_components > X.shape[-1]:
            self.n_components = X.shape[-1]
        if self.whichPCA == 'sklearn':
            self.PCA_sklearn(  X  )  
        else:
            self.PCA_faiss( X )
        return self
            
    def transform(self, X):
        S = (list)(X.shape)
        S[-1] = -1
        X = X.reshape(  -1, X.shape[-1]  )
        if self.whichPCA == 'sklearn':
            tX = np.dot(X, self.Kernels.T) # transform#self.PCA.transform(X)  
        else:
            tX = self.PCA.apply_py(X)
        return tX.reshape(S)

    def inverse_transform(self, X):
        S = (list)(X.shape)
        S[-1] = -1
        X = X.reshape(  -1, X.shape[-1]  )
        if self.whichPCA == 'sklearn':
            tX = np.dot(X, self.Kernels) #self.PCA.inverse_transform(X)
        else:
            tX = self.PCA.reverse_transform(X)
        tX = tX.reshape(S)
        return tX
       
