import numpy as np 
import faiss
import os
import math
from multiprocessing import Process
from core.util.myKMeans import *
from core.util import load_pkl, write_pkl
# in *.data directly save the vectors
# assume vectors all 2D

def Cpredict(X, cent=None, returnDist=False):
    X = np.ascontiguousarray(X.astype('float32'))
    index = faiss.IndexFlatL2(cent.shape[1]) 
    index.add(cent)             
    d, I = index.search(X, 1)
    if returnDist == True:
        return d.reshape(-1)
    return I.reshape(-1)
VERBOSE = True
# from sys import getsizeof
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

    def init_one_cent(self, centID, root, n_file):
        cand_cent, max_dst = None, -1
        for fileID in range(n_file):
            X = load_pkl(root+'/'+str(fileID)+'.data')
            ndst = self.Cpredict(X, np.array(self.cluster_centers_[-1]).reshape(1,-1), returnDist=True)
            dst = load_pkl(root+'/'+'cache_mydKMeans/'+str(fileID)+'.dst')
            dst = np.min(np.concatenate([dst.reshape(-1,1), ndst.reshape(-1,1)], axis=1), axis=1).reshape(-1)
            write_pkl(root+'/'+'cache_mydKMeans/'+str(fileID)+'.dst', dst)
            pos = np.argmax(dst)
            # print(np.max(dst))
            if dst[pos] > max_dst:
                max_dst = dst[pos]
                cand_cent = X[pos]
        print('   init %d, max_dist=%f'%(centID+1, max_dst))
        return cand_cent
    
    def init_centroid_KKZ(self, root, n_file, n_jobs):
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
        write_pkl(root+'/'+'cache_mydKMeans/current.cent', cand_cent)
        self.cluster_centers_ = [cand_cent]
        print('   init 0, max_dist=%f'%max_dst)
        for i in range(self.n_clusters-1):
            # cand_cent = self.init_one_cent(i, root, n_file)
            # by separately call the kkz init helps to reduce the swap memory usage
            os.system('python3 dKM_separable.py init current '+str(i)+' '+root+' '+str(n_file)+' '+str(n_jobs))
            cand_cent = load_pkl(root+'/'+'cache_mydKMeans/current.cent')
            self.cluster_centers_.append(cand_cent)
        os.system('rm -rf '+root+'/'+'cache_mydKMeans')
        self.cluster_centers_ = np.ascontiguousarray(np.array(self.cluster_centers_).astype('float32'))
        # print(self.cluster_centers_)
        print('KKZ Finished')

    def update(self, X, label):
        for i in range(self.n_clusters):
            idx = label == i
            if np.sum(idx) < 1:
                continue
            self.sum_vect[i] += np.sum(X[idx].astype('float64'), axis=0)
            self.freq_vect[i] += (float)(np.sum(idx))
            self.mse += np.sum(np.square(X[idx].astype('float64')-self.cluster_centers_[i].astype('float64'))) # for early stop, Sum of Absolute Difference

    def update_centroid(self):
        # print('min freq',np.min(self.freq_vect))
        self.cluster_centers_ = self.sum_vect / self.freq_vect.reshape(-1,1)
        if np.max(self.cluster_centers_) == math.inf:
            print('Overflow')

    def assign(self, root, n_file):
        for fileID in range(n_file):
            X = load_pkl(root+'/'+str(fileID)+'.data')
            label = self.Cpredict(X)
            self.update(X, label)
        
    def early_stop_checker(self):
        if VERBOSE == True:
            print('iter=%d, sum_square_error=%f'%(self.iter, self.mse))
        if self.last_mse > 0:
            assert self.mse < self.last_mse, 'converge error'
        if np.abs(self.mse-self.last_mse) < 1:
            return True
        self.last_mse = self.mse
        self.mse = float(0.)
        return False

    def fit(self, root, n_file, n_jobs=12):
        if self.stop == True:
            if self.KM is None:
                if VERBOSE == True:
                    print('<INFO> Stopped, more iteration would not help!')
                self.KM = myKMeans(-1).fit(None, self.cluster_centers_)
            return self
        self.clear()
        self.iter += 1
        if self.iter == 0:
            self.init_centroid_KKZ(root, n_file, n_jobs=n_jobs)
        else:
            if n_jobs == 1:
                self.assign(root, n_file)
            else:
                write_pkl(root+'/state_tempelte.state', {'sum_vect':self.sum_vect, 
                                                         'freq_vect':self.freq_vect,
                                                         'mse':self.mse,
                                                         'n_clusters':self.n_clusters})
                multiprocess_dkm(n_jobs, root, n_file, self.cluster_centers_)
                dt = load_pkl(root+'/state_sum.state')
                self.sum_vect += dt['sum_vect']
                self.freq_vect += dt['freq_vect']
                self.mse += dt['mse']
            self.update_centroid()
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
    

def one_process_dkm(root, n_file, start_fileID, cent, processID):
    d = load_pkl(root+'/state_tempelte.state')
    for fileID in range(start_fileID, start_fileID+n_file):
        X = load_pkl(root+'/'+str(fileID)+'.data')
        label = Cpredict(X, cent)
        for i in range(d['n_clusters']):
            idx = label == i
            if np.sum(idx) < 1:
                continue
            d['sum_vect'][i] += np.sum(X[idx].astype('float64'), axis=0)
            d['freq_vect'][i] += (float)(np.sum(idx))
            d['mse'] += np.sum(np.square(X[idx].astype('float64')-cent[i].astype('float64'))) # for early stop, Sum of Absolute Difference
    write_pkl(root+'/state_'+str(processID)+'.state', d)


def multiprocess_dkm(n_jobs, root, n_file, cent):
    n_jobs = np.min([n_jobs, n_file, os.cpu_count()])
    n_files_per_task = n_file // n_jobs +1
    print('n_jobs',n_jobs, n_files_per_task)
    assert n_files_per_task*n_jobs >= n_file, 'not all files are processed'
    p_pool = []
    for start_fileID in range(n_jobs):
        p = Process(target=one_process_dkm, args=(root, min(n_files_per_task, n_file-start_fileID*n_files_per_task), start_fileID*n_files_per_task, cent, start_fileID, ))
        p_pool.append(p)
    for i in range(n_jobs):
        p_pool[i].start()
        p_pool[i].join()
    # aggregate
    d = load_pkl(root+'/state_tempelte.state')
    for processID in range(n_jobs):
        dt = load_pkl(root+'/state_'+str(processID)+'.state')
        os.system('rm -rf '+root+'/state_'+str(processID)+'.state')
        d['sum_vect'] += dt['sum_vect']
        d['freq_vect'] += dt['freq_vect']
        d['mse'] += dt['mse']
    write_pkl(root+'/state_sum.state', d)





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