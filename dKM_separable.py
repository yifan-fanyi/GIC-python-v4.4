import numpy as np 
import faiss
import os
import math
from core.util.myKMeans import *
from core.util import load_pkl, write_pkl
import sys
from multiprocessing import Process

def Cpredict(X, cent=None, returnDist=False):
    X = np.ascontiguousarray(X.astype('float32'))
    index = faiss.IndexFlatL2(cent.shape[1]) 
    index.add(cent)             
    d, I = index.search(X, 1)
    if returnDist == True:
        return d.reshape(-1)
    return I.reshape(-1)


def one_process_dkm_kkz(centfile, root, n_file, start_fileID, processID):
    cand_cent, max_dst = None, -1
    for fileID in range(start_fileID, start_fileID+n_file):
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
    write_pkl(root+'/'+'cache_mydKMeans/'+centfile+'_'+str(processID)+'.cent', [max_dst, cand_cent])
    print(max_dst, processID)
    # return cand_cent

def init_one_cent(centfile, centID, root, n_file, n_jobs):
    n_jobs = np.min([n_jobs, n_file, os.cpu_count()])
    # print('n_jobs',n_jobs)
    n_files_per_task = n_file // n_jobs +1
    p_pool = []
    for start_fileID in range(n_jobs):
        p = Process(target=one_process_dkm_kkz, args=(centfile, root, min(n_files_per_task, n_file-start_fileID*n_files_per_task), start_fileID*n_files_per_task, start_fileID, ))
        p_pool.append(p)
    for i in range(n_jobs):
        p_pool[i].start()
        p_pool[i].join()
    # aggregate
    dst, cent = 0, None
    for processID in range(n_jobs):
        d = load_pkl(root+'/'+'cache_mydKMeans/'+centfile+'_'+str(processID)+'.cent')
        if d[0] > dst:
            dst = d[0]
            cent = d[1]
    write_pkl(root+'/'+'cache_mydKMeans/'+centfile+'.cent', cent)
    print('   init %d, max_dist=%f'%(centID+1, dst))
    
def init_one_cent_s(centfile, centID, root, n_file):
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
if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == 'init':
        centfile, centID, root, n_file, n_jobs = sys.argv[2], (int)(sys.argv[3]), sys.argv[4], (int)(sys.argv[5]), (int)(sys.argv[6])
        # print('n_jobs', n_jobs)
        if n_jobs == 1:
            init_one_cent_s(centfile, centID, root, n_file)
        else:
            init_one_cent(centfile, centID, root, n_file, n_jobs)
