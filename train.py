import numpy as np
from core.rdvq1ac_noT import *
from core.Distributed_KMeans import *
import os
import pickle

# root: root path
# X: some smaples, will be used to train entropy coding and find the skip_th to derive input for next level
# n_files, residual files, the input will locates in root/p_win/ files are named as 0.data 1.data ...
#   
def train_one_vq(root, X, win, p_win, n_clusters, n_file, Lagrange_multip):
    file_list = []
    for i in range(n_file):
        file_list.append(str(i)+'.data')
    try:
        with open(root+'/model/dkm'+str(X.shape[1])+'_'+str(win)+'.model','rb') as f:
            dkm = pickle.load(f)
    except:
        os.system('mkdir '+root+'/'+str(X.shape[1])+'_'+str(win))
        dkm = Distributed_KMeans(n_clusters=n_clusters, 
                                size=X.shape[1], 
                                win=win, 
                                datatype="SHORT", 
                                frame_each_file=200, 
                                n_frames=100).fit(folder=root+'/'+str(X.shape[1])+'_'+str(p_win), 
                                                file_list=file_list)
        with open(root+'/model/dkm_'+str(X.shape[1])+'_'+str(win)+'.model','wb') as f:
            pickle.dump(dkm, f, 4)
    try:
        with open(root+'/model/vq'+str(X.shape[1])+'_'+str(win)+'.model','rb') as f:
            vq = pickle.load(f)
    except:
        vq = VQ_noT(n_clusters_list=[[n_clusters]], win_list=[win], n_dim_list=[[win**2*3]], Lagrange_multip=Lagrange_multip, acc_bpp=0)
        vq.fit(dkm, X)
        with open(root+'/model/vq_'+str(X.shape[1])+'_'+str(win)+'.model','wb') as f:
            pickle.dump(vq, f, 4)

    dkm.predict(root+'/'+str(X.shape[1])+'_'+str(p_win), 
                file_list, 
                root+'/'+str(X.shape[1])+'_'+str(win), 
                th=vq.th*1.1)
    return vq.predict(X)


if __name__ == "__main__":
    from core.data import *
    t = load('test', 144, size=[256])
    X = t[0]
    print(X.shape)
    train_one_vq(root='/Users/alex/Desktop/test_data/', 
                 X=X, 
                 win=4, 
                 p_win=8, 
                 n_clusters=16, 
                 n_file=2, 
                 Lagrange_multip=1000)
