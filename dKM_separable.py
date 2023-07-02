import numpy as np 
import faiss
import os
import math
from core.util.myKMeans import *
from core.util import load_pkl, write_pkl
from core.util.Huffman import *
from core.cwSaab import cwSaab
from core.util.myKMeans import myKMeans
from core.util.mydKMeans import mydKMeans
import math
from core.util.Huffman import Huffman
import numpy as np
from core.util import Time, myLog, Shrink, load_pkl, write_pkl
from core.util.ac import HierarchyCABAC, BAC
from core.util.evaluate import MSE
from core.util.ReSample import *
from core.VQEntropy import VQEntropy

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
    # print(max_dst, processID)
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
    for i in range(n_jobs):
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
    # print('n_jobs',n_jobs, 'n_files_per_task',n_files_per_task)
    assert n_files_per_task*n_jobs >= n_file, 'not all files are processed'
    p_pool = []
    for start_fileID in range(n_jobs):
        p = Process(target=one_process_dkm, args=(root, min(n_files_per_task, n_file-start_fileID*n_files_per_task), start_fileID*n_files_per_task, cent, start_fileID, ))
        p_pool.append(p)
    for i in range(n_jobs):
        p_pool[i].start()
    for i in range(n_jobs):
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

def one_process_rd(root, n_file, start_fileID, level, pos, h):
    vq = load_pkl(root+'/current.vq')
    for fileID in range(start_fileID, start_fileID+n_file):
        vq.isdistributed = [fileID, root, -1]
        X = load_pkl(root+'/'+str(fileID)+'.iR')
        tX = load_pkl(root+'/'+str(fileID)+'.cwsaab')
        iX = vq.RD_search_km(tX, X, level, pos, None, False)
        X[:,:,:,:vq.n_dim_list[level][pos]] -= iX[:, :,:,:vq.n_dim_list[level][pos]]
        write_pkl(root+'/'+str(fileID)+'.iR', X)
    write_pkl(root+'/tmp_current_'+str(h)+'.vq', vq)
    # return cand_cent

def process_rd(root, n_file, level, pos, n_jobs):
    n_jobs = np.min([n_jobs, n_file, os.cpu_count()])
    # print('n_jobs',n_jobs)
    n_files_per_task = n_file // n_jobs +1
    p_pool = []
    for start_fileID in range(n_jobs):
        p = Process(target=one_process_rd, args=(root, min(n_files_per_task, n_file-start_fileID*n_files_per_task), start_fileID*n_files_per_task, level, pos, start_fileID, ))
        p_pool.append(p)
    for i in range(n_jobs):
        p_pool[i].start()
    for i in range(n_jobs):
        p_pool[i].join()
    vq = load_pkl(root+'/current.vq')
    def merge_dict(d, dd):
        for k in dd:
            d[k] = dd[k]
        return d
    for i in range(n_jobs):
        vqt = load_pkl(root+'/tmp_current_'+str(i)+'.vq')
        vq.max_dmse = merge_dict(vq.max_dmse, vqt.max_dmse)
        vq.skip_th_range = merge_dict(vq.skip_th_range, vqt.skip_th_range)
        vq.Huffman = merge_dict(vq.Huffman, vqt.Huffman)
    write_pkl(root+'/current.vq', vq)
    

def one_process_vqentropy(root, n_file, myhash, kmidx):
    vq = load_pkl(root+'/current.vq')
    
    for i in kmidx:
        d = {}
        vq.isdistributed[2] = i
        vq.skip_th_range[myhash+'_'+str(vq.isdistributed[2])] = np.log2(np.max(vq.isdistributed[2])) / 80
        km = vq.myKMeans[myhash][vq.isdistributed[2]]
        nc = km.n_clusters
        d[myhash+'_'+str(vq.isdistributed[2])+'_h'] = Huffman().fit_distributed(root+'/kmidx_'+str(i)+'/', n_file, nc)
        # print(self.max_dmse)
        if np.max(km.inverse_predict(np.arange(nc).reshape(-1, 1))) == math.inf:
            print('Overflow', km.inverse_predict(np.arange(nc).reshape(-1, 1)))
        c = VQEntropy(nc, km.inverse_predict(np.arange(nc).reshape(-1, 1)))
        c.fit_distributed(root+'/kmidx_'+str(i)+'/', 
                                                                                                                                              n_file, skrange=vq.max_dmse[myhash+'_'+str(vq.isdistributed[2])])
#                     continue
        d[myhash+'_'+str(vq.isdistributed[2])] = c
        write_pkl(root+'/'+myhash+'_'+str(i)+'.ec', d)
        # print('wrote '+root+'/'+myhash+'_'+str(i)+'.ec')

# schedule the job based on n_clusters
# linear relationship, if n_jobs=2, and the candidates kmeans with codewords 1024, 512, 256, 256
# one job will be used to train 1024's ec
# other three's ec will be trained by second job
def process_vqqentropy(root, n_file, myhash, n_jobs):
    vq = load_pkl(root+'/current.vq')
    km_list = vq.myKMeans[myhash]
    n_km = len(km_list)
    n_jobs = np.min([n_jobs, n_km, os.cpu_count()])
    kmidx, ct = [], np.zeros(n_jobs)
    for i in range(n_jobs):
        kmidx.append([])
    for i in range(n_km):
        nc = km_list[i].n_clusters
        pos = np.argmin(ct)
        kmidx[pos].append(i)
        ct[pos] += nc
    p_pool = []
    for i in range(n_jobs):
        p = Process(target=one_process_vqentropy, args=(root, n_file, myhash, kmidx[i], ))
        p_pool.append(p)
    for i in range(n_jobs):
        p_pool[i].start()
    for i in range(n_jobs):
        p_pool[i].join()



if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == 'init':
        centfile, centID, root, n_file, n_jobs = sys.argv[2], (int)(sys.argv[3]), sys.argv[4], (int)(sys.argv[5]), (int)(sys.argv[6])
        # print('n_jobs', n_jobs)
        if n_jobs == 1:
            init_one_cent_s(centfile, centID, root, n_file)
        else:
            init_one_cent(centfile, centID, root, n_file, n_jobs)
    if opt == 'fit':
        n_jobs, root, n_file = (int)(sys.argv[2]), sys.argv[3], (int)(sys.argv[4])
        cent = load_pkl(root+'/cent.pkl')
        multiprocess_dkm(n_jobs, root, n_file, cent)
    if opt == 'vqentropy':
        root, n_file, myhash, n_jobs = sys.argv[2], (int)(sys.argv[3]), sys.argv[4], (int)(sys.argv[5])
        process_vqqentropy(root, n_file, myhash, n_jobs)
    if opt == 'rd':
        root, n_file, level, pos, n_jobs = sys.argv[2], (int)(sys.argv[3]), (int)(sys.argv[4]), (int)(sys.argv[5]), (int)(sys.argv[6])
        process_rd(root, n_file, level, pos, n_jobs)