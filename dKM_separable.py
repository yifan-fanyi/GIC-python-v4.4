import numpy as np 
import faiss
import os
import math
from core.util.myKMeans import *
from core.util import load_pkl, write_pkl
import sys

def Cpredict(X, cent=None, returnDist=False):
    X = np.ascontiguousarray(X.astype('float32'))
    index = faiss.IndexFlatL2(cent.shape[1]) 
    index.add(cent)             
    d, I = index.search(X, 1)
    if returnDist == True:
        return d.reshape(-1)
    return I.reshape(-1)

def init_one_cent(centfile, centID, root, n_file):
    cand_cent, max_dst = None, -1
    for fileID in range(n_file):
        X = load_pkl(root+'/'+str(fileID)+'.data')
        cent = load_pkl(root+'/'+'cache_mydKMeans/'+centfile+'.cent')
        ndst = Cpredict(X, np.array(cent).reshape(1,-1), returnDist=True)
        dst = load_pkl(root+'/'+'cache_mydKMeans/'+str(fileID)+'.dst')
        dst = np.min(np.concatenate([dst.reshape(-1,1), ndst.reshape(-1,1)], axis=1), axis=1).reshape(-1)
        write_pkl(root+'/'+'cache_mydKMeans/'+str(fileID)+'.dst', dst)
        pos = np.argmax(dst)
        # print(np.max(dst))
        if dst[pos] > max_dst:
            max_dst = dst[pos]
            cand_cent = X[pos]
    print('   init %d, max_dist=%f'%(centID+1, max_dst))
    write_pkl(root+'/'+'cache_mydKMeans/'+centfile+'.cent', cand_cent)
    return cand_cent

def update(X, label, d):
    for i in range(self.n_clusters):
        idx = label == i
        if np.sum(idx) < 1:
            continue
        d['sum_vect'][i] += np.sum(X[idx].astype('float64'), axis=0)
        d['freq_vect'][i] += (float)(np.sum(idx))
        d['mse'] += np.sum(np.square(X[idx].astype('float64')-self.cluster_centers_[i].astype('float64'))) # for early stop, Sum of Absolute Difference
    return d
def assign(root, n_file, progressfile):
    d = load_pkl(root+'/'+'cache_mydKMeans/'+progressfile+'.progress')
    for fileID in range(n_file):
        X = load_pkl(root+'/'+str(fileID)+'.data')
        label = Cpredict(X, d['cent'])
        d = update(X, label, d)
    d['cent'] = d['sum_vect'] / d['freq_vect'].reshape(-1,1)
    write_pkl(root+'/'+'cache_mydKMeans/'+progressfile+'.progress', d)
    
if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == 'init':
        centfile, centID, root, n_file = sys.argv[2], (int)(sys.argv[3]), sys.argv[4], (int)(sys.argv[5])
        init_one_cent(centfile, centID, root, n_file)
    if opt == 'fit':# fit seems not increase the memory dramatically, this part will not be used now
        root, n_file, progressfile = sys.argv[2], (int)(sys.argv[3]), sys.argv[4]
        assign(root, n_file, progressfile)