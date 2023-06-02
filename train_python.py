from core.rdvq1ac import VQ
import numpy as np
from core.util import *
from core.util.evaluate import MSE
import json
import os
from core.util.ReSample import resize
# .iR residual, will be keep updated
f = open('./par/par_unit.json',)
par = json.load(f)

def vq_one_resolution(root, n_file, par):
    
    n_clusters_list, n_dim_list = [],[]
    for win in par['level_list']:
        n_clusters_list += [par['n_clusters_list-'+str(win)]]
        n_dim_list += [par['n_dim_list-'+str(win)]]

    vq = VQ(n_clusters_list=n_clusters_list, 
            win_list=par['win_list'], 
            n_dim_list=n_dim_list, 
            transform_split=par['transform_split'],
            Lagrange_multip=par['Lagrange_multip'], acc_bpp=0)
    vq.fit_distributed(root, n_file)
    return vq


def run(par, run_root, data_root, n_file):
    vq = {}
    resolution_list = par['resolution_list']
    for i in range(len(resolution_list)):
        os.system('mkdir '+run_root+'/'+str(resolution_list[i]))
        if par['id'] == resolution_list[0]: 
            for fileID in range(n_file):
                X = load_pkl(data_root+'/'+str(resolution_list[0])+'/'+str(fileID)+'.spatial_raw')
                write_pkl(run_root+'/'+str(resolution_list[0])+'/'+str(fileID)+'.iR', X)        
        else:
            for fileID in range(n_file):
                iR = load_pkl(run_root+'/'+str(resolution_list[i-1])+'/'+str(fileID)+'.iR')
                X = load_pkl(data_root+'/'+str(resolution_list[i-1])+'/'+str(fileID)+'.spatial_raw')
                iX = resize(X-iR, resolution_list[i])
                X = load_pkl(run_root+'/'+str(resolution_list[i])+'/'+str(fileID)+'.iR')
                write_pkl(X-iX, run_root+'/'+str(resolution_list[i])+'/'+str(fileID)+'.iR')
        vq[str(resolution_list[i])] = vq_one_resolution(run_root+'/'+str(resolution_list[i])+'/', 
                          n_file, 
                          par[str(resolution_list[i])])
    return vq

