import numpy as np 
import faiss
import os
from core.util.myKMeans import *
from core.util import load_pkl, write_pkl
# in *.data directly save the vectors
# assume vectors all 2D
VERBOSE = True

class mydKMeans:
    def __init__(self, n_clusters, dim):
        self.n_clusters = n_clusters
        self.dim = dim
        self.sum_vect = np.zeros((n_clusters, dim), dtype='float64')
        self.freq_vect = np.zeros((n_clusters), dtype='int64')
        self.cluster_centers_ = np.zeros((n_clusters, dim), dtype='float64')
        self.iter = -1 # number of iteration
        self.mse = float(0.) # sum of the absolute difference
        self.last_mse = float(-1.) # mse of last lteration
        self.stop = False # early stop flag
        self.KM = None

    def clear(self):
        self.sum_vect = np.zeros((self.n_clusters, self.dim), dtype='float64')
        self.freq_vect = np.zeros((self.n_clusters), dtype='int64')

    def Cpredict(self, X, cent=None, returnDist=False):
        X = np.ascontiguousarray(X.astype('float32'))
        if cent is None:
            cent = np.ascontiguousarray(self.cluster_centers_.astype('float32'))
        index = faiss.IndexFlatL2(cent.shape[1]) 
        index.add(cent)             
        d, I = index.search(X, 1)
        if returnDist == True:
            return d.reshape(-1)
        return I.reshape(-1)

    def inverse_Cpredict(self, label):
        return self.cluster_centers_[label]

    def init_centroid_KKZ(self, root, n_file):
        s, c = np.zeros(self.dim), 0
        for fileID in range(n_file):
            X = load_pkl(root+'/'+str(fileID)+'.data')
            s += np.mean(X,axis=0)
            c += 1
        X0 = s/c
        cand_cent, max_dst = None, -1
        os.system('mkdir '+root+'cache_mydKMeans')
        for fileID in range(n_file):
            X = load_pkl(root+'/'+str(fileID)+'.data')
            dst = self.Cpredict(X, np.array(X0).reshape(1, -1), returnDist=True)
            write_pkl(root+'/'+'cache_mydKMeans/'+str(fileID)+'.dst', dst)
            pos = np.argmax(dst.reshape(-1))
            if dst[pos] > max_dst:
                max_dst = dst[pos]
                cand_cent = X[pos]
        self.cluster_centers_ = [cand_cent]
        for _ in range(self.n_clusters-1):
            cand_cent, max_dst = None, -1
            for fileID in range(n_file):
                X = load_pkl(root+'/'+str(fileID)+'.data')
                dst = self.Cpredict(X, np.array(self.cluster_centers_), returnDist=True)
                pos = np.argmax(dst.reshape(-1))
                # print(np.max(dst))
                if dst[pos] > max_dst:
                    max_dst = dst[pos]
                    cand_cent = X[pos]
            self.cluster_centers_.append(cand_cent)
        os.system('rm -rf '+root+'/'+'cache_mydKMeans')
        self.cluster_centers_ = np.ascontiguousarray(np.array(self.cluster_centers_).astype('float32'))
        # print(self.cluster_centers_)
    def update(self, X, label):
        for i in range(self.n_clusters):
            idx = label == i
            if np.sum(idx) < 1:
                continue
            self.sum_vect[i] += np.sum(X[idx].astype('float64'), axis=0)
            self.freq_vect[i] += (float)(np.sum(idx))
            self.mse += np.sum(np.square(X[idx].astype('float64')-self.cluster_centers_[i].astype('float64'))) # for early stop, Sum of Absolute Difference

    def update_centroid(self):
        self.cluster_centers_ = self.sum_vect / self.freq_vect.reshape(-1,1)

    def assign(self, root, n_file):
        for fileID in range(n_file):
            X = load_pkl(root+'/'+str(fileID)+'.data')
            label = self.Cpredict(X)
            self.update(X, label)
        self.update_centroid()
    
    def early_stop_checker(self):
        if VERBOSE == True:
            print('iter=%d, sum_square_error=%f'%(self.iter, self.mse))
        if np.abs(self.mse-self.last_mse) < 1e-3:
            return True
        self.last_mse = self.mse
        self.mse = float(0.)
        return False

    def fit(self, root, n_file):
        if self.stop == True:
            if self.KM is None:
                if VERBOSE == True:
                    print('<INFO> Stopped, more iteration would not help!')
                    self.KM = myKMeans(-1).fit(None, self.cluster_centers_)
            return self
        self.clear()
        self.iter += 1
        if self.iter == 0:
            self.init_centroid_KKZ(root, n_file)
        else:
            self.assign(root, n_file)
            self.stop = self.early_stop_checker()
        return self
    
    def predict_distributed(self, root, n_file, dst_root):
        for fileID in range(n_file):
            X = load_pkl(root+'/'+str(fileID)+'.data')
            label = self.predict(X)
            iX = self.inverse_predict(label)
            write_pkl(dst_root+'/'+str(fileID)+'.label', label)
            # save residual
            write_pkl(dst_root+'/'+str(fileID)+'.data', X-iX)

    def inverse_predict_distributed(self, root, n_file, dst_root):
        pass
        # no need, currently
        
    def predict(self, X):
        return self.KM.predict(X)
    def inverse_predict(self, label):
        return self.KM.inverse_predict(label)
    
if __name__ == "__main__":
    from core.util.evaluate import MSE
    def entropy(x, nbin):
        p = np.zeros((nbin))
        x = x.reshape(-1).astype('int32')
        for i in range(len(x)):
            p[x[i]] +=1.
        p = p/np.sum(p)
        return -np.sum(p * np.log2(p+1e-10))
        
    def RMSE(x, y):
        return np.sqrt(MSE(x,y))
    from core.util.myKMeans import *
    from core.data import *
    from core.util import Shrink, invShrink
    
    try: 
        X1 = load_pkl('0.data')
        X2 = load_pkl('1.data')
        # print(X1.shape, np.min(X1), np.max(X1))
    except:
        print('File not found')
        tmp = load(Rtype='train', ct=[50,0], size=[256])
        Y256 = tmp[0]
        print(Y256.shape)
        win = 4
        X1 = Shrink(Y256[:25], win=4)
        X1 = X1.reshape(-1, X1.shape[-1])
        X2 = Shrink(Y256[25:], win=4)
        X2 = X2.reshape(-1, X2.shape[-1])
        write_pkl('0.data', X1)
        write_pkl('1.data', X2)

    nc = 128
    for nc in [8,16,32,64,128,256,512,1024,2048]:
        dkm = mydKMeans(nc, 4**2*3)
        for i in range(20000):
            dkm.fit('./', 2)
        
        print('<INFO> nc=%d'%nc)
        X = np.concatenate([X1, X2], axis=0)
        label = dkm.predict(X)
        iX = dkm.inverse_predict(label)
        print('   <INFO> dkm MSE=%f H=%f'%(MSE(X,iX), entropy(label, nc)))
        km = myKMeans(nc, fast=True).fit(X)
        label = km.predict(X)
        iX = km.inverse_predict(label)
        print('   <INFO> faiss MSE=%f H=%f'%(MSE(X,iX), entropy(label, nc)))


'''
performance on training 
nc, dkm mse, dkm h, faiss mse, faiss h
[[8,719.965719,2.908262,721.194230,2.910363],
[16,547.784992,3.726811,547.450462,3.739569],
[32,422.213132,4.483393,420.692881,4.710556],
[64,334.394180,5.373568,328.253915,5.629740],
[128,277.523840,6.169170,272.651698,6.597785],
[256,229.308168,7.004706,225.437888,7.549480],
[512,196.057496,7.654622,190.178135,8.508877],
[1024,167.461159,8.361140,162.555546,9.475010]]
'''