from core.rdvq1ac import VQ
import numpy as np
from core.util import *
from core.util.evaluate import MSE
import json
import os
from core.util.ReSample import resize
# .iR residual, will be keep updated


def vq_one_resolution(root, n_file, par):
    
    n_clusters_list, n_dim_list = [],[]
    for win in par['level_list']:
        n_clusters_list += [par['n_cluster_list-'+str(win)]]
        n_dim_list += [par['n_dim_list-'+str(win)]]

    vq = VQ(n_clusters_list=n_clusters_list, 
            win_list=par['win_list'], 
            n_dim_list=n_dim_list, 
            transform_split=par['transform_split'],
            Lagrange_multip=par['Lagrange_multip'], acc_bpp=0)
    vq.fit_distributed(root, n_file)
    return vq


def run(gpar, run_root, data_root, n_file):
    vq = {}
    resolution_list = gpar['resolution_list']
    for i in range(len(resolution_list)):
        print('fit', resolution_list[i])
        par = gpar[str(resolution_list[i])]
        os.system('mkdir '+run_root+'/'+str(resolution_list[i]))
        if par['id'] == resolution_list[0]: 
            for fileID in range(n_file):
                X = load_pkl(data_root+'/'+str(resolution_list[0])+'/'+str(fileID)+'.spatial_raw').astype('float32')-128
                write_pkl(run_root+'/'+str(resolution_list[0])+'/'+str(fileID)+'.spatial_data', X)        
        else:
            for fileID in range(n_file):
                iR = load_pkl(run_root+'/'+str(resolution_list[i-1])+'/'+str(fileID)+'.iRspatial').astype('float32')
                X = load_pkl(data_root+'/'+str(resolution_list[i-1])+'/'+str(fileID)+'.spatial_raw').astype('float32')-128
                iX = resize(X-iR, resolution_list[i])
                X = load_pkl(data_root+'/'+str(resolution_list[i])+'/'+str(fileID)+'.spatial_raw').astype('float32')-128
                write_pkl(run_root+'/'+str(resolution_list[i])+'/'+str(fileID)+'.spatial_data', X-iX)
        vq[str(resolution_list[i])] = vq_one_resolution(run_root+'/'+str(resolution_list[i])+'/', 
                          n_file, 
                          par)
        with open(run_root+'/vq_'+str(resolution_list[i])+'.model', 'wb') as f:
            pickle.dump(vq[str(resolution_list[i])], f, 4)
    return vq

def predict(vq_list, resolution_list, X_list):
    for i in range(len(resolution_list)):
        print('predict', resolution_list[i])
        vq = vq_list[str(resolution_list[i])]
        X = X_list[str(resolution_list[i])]
        if i == 0:
            iR = vq.predict(X)
        else:
            iR = X - resize(X_list[str(resolution_list[i-1])]-iR, X.shape[1]) 
            iR = vq.predict(X)
    return iR
    return X_list[str(resolution_list[-1])] -iR

def predict_distributed(vq_list, resolution_list, root, n_file):
    os.system('mkdir '+root+'/predict'+str(resolution_list[-1]))
    for fileID in range(n_file):
        X_list = []
        for i in range(len(resolution_list)):
            X_list.append(load_pkl(root+'/'+str(resolution_list[i])+'/'+str(fileID)+'.spatial_raw'))
        iR = predict(vq_list, resolution_list, X_list)
        write_pkl(root+'/predict'+str(resolution_list[-1])+'/'+str(fileID)+'.outiR', iR)

def predict_resume(vq_list, resolution_list, X_list, iR):
    # resolution_list[0] inital_resolution, already performed VQ
    for i in range(1, resolution_list):
        vq = vq_list[str(resolution_list[i])]
        X = X_list[str(resolution_list[i])]
        iR = X - resize(X_list[str(resolution_list[i-1])]-iR, X.shape[1]) 
        iR = vq.predict(X)
    return iR

def predict_resume_distributed(vq_list, resolution_list, root, n_file):
    os.system('mkdir '+root+'/predict'+str(resolution_list[-1]))
    for fileID in range(n_file):
        X_list = []
        iR = load_pkl(root+'/predict'+str(resolution_list[0])+'/'+str(fileID)+'.outiR')
        for i in range( len(resolution_list)):
            X_list.append(load_pkl(root+'/'+str(resolution_list[i])+'/'+str(fileID)+'.spatial_raw'))
        iR = predict_resume(vq_list, resolution_list, X_list, iR)
        write_pkl(root+'/predict'+str(resolution_list[-1])+'/'+str(fileID)+'.outiR', iR)

if __name__ == "__main__":
    # X = load_pkl('./unit/data/'+'/'+str(256)+'/'+str(0)+'.spatial_raw')
    # print(np.sum(X))
    f = open('./par/par.json',)
    par = json.load(f)
    vq = run(par, run_root='/Users/alex/Desktop/unit/run/', data_root='/Users/alex/Desktop/unit/data/', n_file=11)
    with open('model.pkl','wb') as f:
        pickle.dump(vq, f)