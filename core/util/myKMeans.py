# 2021.05.10
# change inv_predict to inverse_perdict
# https://www.kdnuggets.com/2021/01/k-means-faster-lower-error-scikit-learn.html
# faster kmeans
# gpu is supported
# conda install -c conda-forge faiss-gpu
#
import faiss
import numpy as np
import copy
import sklearn
from sklearn import cluster
from sklearn.mixture import GaussianMixture
print('verbose=True')

def Cpredict(X, cent, returnDist=False):
    X = np.ascontiguousarray(X.astype('float32'))
    cent = np.ascontiguousarray(cent.astype('float32'))
    index = faiss.IndexFlatL2(cent.shape[1]) 
    index.add(cent)             
    d, I = index.search(X, 1)
    if returnDist == True:
        return d
    return I

def KKZ_init(X, n_clusters):
    X = X.reshape(-1, X.shape[-1])
    X0 = np.mean(X,axis=0)
    dst = Cpredict(X, np.array(X0).reshape(1, -1), returnDist=True)
    cluster_centers = [X[np.argmax(dst.reshape(-1))]]
    for _ in range(n_clusters-1):
        ndst = Cpredict(X, np.array(cluster_centers[-1]).reshape(1, -1), returnDist=True)
        dst = np.min(np.concatenate([dst, ndst], axis=1), axis=1).reshape(-1, 1)
        cluster_centers.append(X[np.argmax(dst.reshape(-1))])
    return np.ascontiguousarray(np.array(cluster_centers).astype('float32'))

class fast_KMeans:
    def __init__(self, n_clusters=8, n_init=1, max_iter=300, gpu=False, n_threads=24, KKZinit=False):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.gpu = False     
        self.KKZinit = KKZinit
        faiss.omp_set_num_threads(n_threads)       
        self.__version__ = faiss.__version__

    def fit(self, X):
        verbose = False
        if self.gpu != False:
            self.kmeans = faiss.Kmeans(d=X.shape[1],
                                    k=self.n_clusters,
                                    niter=self.max_iter,
                                    nredo=self.n_init,
                                    gpu=self.gpu,
                                    min_points_per_centroid=39,
                                    max_points_per_centroid=1024,
                                    verbose=verbose)
        else:
            self.kmeans = faiss.Kmeans(d=X.shape[1],
                                    k=self.n_clusters,
                                    niter=self.max_iter,
                                    nredo=self.n_init,
                                    min_points_per_centroid=39,
                                    max_points_per_centroid=1024,
                                    verbose=verbose)
        X = np.ascontiguousarray(X.astype('float32'))
        if self.KKZinit == True:
            self.kmeans.centroids = KKZ_init(X, self.n_clusters)
            #print('finish kkz')
        self.kmeans.train(X)
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]
        return self

def sort_by_eng(Cent):
    eng = np.sum(np.square(Cent), axis=1)
    idx = np.argsort(eng)
    mp, imp = {}, {}
    for i in range(len(idx)):
        assert (i not in mp.keys()), "Err"
        assert (idx[i] not in imp.keys()), 'err'
        mp[i] = idx[i]
        imp[idx[i]] = i
    return mp, imp

def sort_by_hist(x, bins):
    def Hist(x, bins):
        hist = np.zeros(bins)
        x = x.reshape(-1)
        for i in x:
            hist[i] += 1
        return hist
    hist = Hist(x, bins)
    idx = np.argsort(hist)[::-1]
    mp, imp = {}, {}
    for i in range(len(hist)):
        mp[idx[i]] = i
        imp[i] = idx[i]
    return mp, imp

class Mapping():
    def __init__(self, mp, imp):
        self.map, self.inv_map = mp, imp
        self.version = '2021.05.14'

    def transform(self, label):
        S = label.shape
        label = label.reshape(-1)
        for i in range(len(label)):
            label[i] = self.map[label[i]]
        return label.reshape(S)
    
    def inverse_transform(self, l):
        S = l.shape
        label = copy.deepcopy(l).reshape(-1)
        for i in range(len(label)):
            label[i] = self.inv_map[label[i]]
        return label.reshape(S)
    
class myKMeans():
    def __init__(self, n_clusters=-1, trunc=-1, fast=True, gpu=False, n_threads=24, sort=True, saveObj=False, KKZinit=False):
        if fast == True:
            self.KM          = fast_KMeans(  n_clusters=n_clusters, n_init=1 , gpu=gpu, n_threads=n_threads, KKZinit=KKZinit)
            self.version_   = self.KM.__version__
        else:
            self.KM          = cluster.KMeans(  n_clusters=n_clusters, n_init=1  )
            self.version_ =  sklearn. __version__ 
        self.n_clusters = n_clusters
        self.cluster_centers_        = []
        self.trunc       = trunc
        self.fast = fast
        self.sort        = sort
        self.saveObj     = saveObj
        self.KKZinit = KKZinit
        self.enable_prob = False
        self.version = '2021.05.31'
        
    def truncate(self, X):
        if self.trunc != -1:
            X[:, self.trunc:] *= 0
        return X
    
    def fit(self, X, cluster_centers=None):
        if cluster_centers is None:
            X = X.reshape(  -1, X.shape[-1]  ).astype('float16')
            self.truncate(X)
            if X.shape[0] > self.n_clusters:
                if self.fast == False and self.KKZinit == True:
                    self.KM          = cluster.KMeans(  n_clusters=self.n_clusters, n_init=1  , init=KKZ_init(X, self.n_clusters))

                self.KM.fit(  X  )
                self.cluster_centers_ = copy.deepcopy(np.array(  self.KM.cluster_centers_  )).astype('float32')
                if self.saveObj == False:
                    self.KM = None
            else:
                self.cluster_centers_ = copy.deepcopy(X).astype('float32')
                self.n_clusters = X.shape[0]
                self.saveObj = False
        else:
            self.cluster_centers_ = cluster_centers
            self.n_clusters = len(cluster_centers)
            self.saveObj = False
            self.sort = False
        if cluster_centers is None:
            if self.saveObj == False:
                l = self.Cpredict(X)
            else:
                l = self.KM.predict(X)
        if self.sort == 1:
            mp, imp = sort_by_hist(l, self.n_clusters)
        if self.sort == 2:
            mp, imp = sort_by_eng(np.array(  self.cluster_centers_  ))
        if self.sort > 0:
            self.MP = Mapping(mp, imp)
        if len(self.cluster_centers_) > 1 and self.enable_prob == True:
            try:
                a = self.MP.transform(np.arange((len(self.cluster_centers_))))
                self.clf = GaussianMixture(n_components=len(self.cluster_centers_), 
                                        covariance_type='spherical', 
                                        means_init=self.cluster_centers_[a],
                                        n_init= 1 ,
                                        max_iter=1).fit(X)
            except:
                self.enable_prob = False
        return self

    def Cpredict(self, X):
        X = np.ascontiguousarray(X.astype('float32'))
        cent = np.ascontiguousarray(self.cluster_centers_.astype('float32'))
        index = faiss.IndexFlatL2(cent.shape[1]) 
        index.add(cent)             
        d, I = index.search(X, 1)
        return I


    def predict(self, X):
        S = (list)(X.shape)
        X = X.astype('float16')
        S[-1] = -1
        X = X.reshape(-1, X.shape[-1])
        if self.saveObj == True:
            idx = self.KM.predict(X)
        else:
            idx = self.Cpredict(X)
        if self.sort == True:
            idx = self.MP.transform(idx)
        return idx.reshape(S)

    def predict_proba(self, X):
        assert self.enable_prob == True, 'myKMeans.predict_proba not available!'
        S = (list)(X.shape)
        X = X.astype('float16')
        S[-1] = -1
        X = X.reshape(-1, X.shape[-1])
        if len(self.cluster_centers_) > 1:
            return self.clf.predict_proba(X).reshape(S)
        else:
            return np.ones((len(X), 1))

    def inverse_predict(self, idx):
        idx = idx.astype('int16')
        S = (list)(idx.shape)
        S[-1] = -1
        idx = idx.reshape(-1,)
        idx[idx >= self.n_clusters] = 0
        if self.sort == True:
            idx = self.MP.inverse_transform(idx)
        X = self.cluster_centers_[idx]
        return X.reshape(S).astype('float16')