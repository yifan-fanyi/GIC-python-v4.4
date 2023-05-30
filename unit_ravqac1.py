from core.rdvq1ac import VQ
import numpy as np
from core.util import *
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
    X1 = load_pkl('./unit/0.spatial_data')
    X2 = load_pkl('./unit/1.spatial_data')
    # print(X1.shape, np.min(X1), np.max(X1))
except:
    print('File not found')
    tmp = load(Rtype='train', ct=[50,0], size=[256])
    Y256 = tmp[0]
    print(Y256.shape)
    write_pkl('./unit/0.spatial_data', Y256[:25])
    write_pkl('./unit/1.spatial_data', Y256[25:])


vq = VQ(n_clusters_list=[[8,8],[8]], win_list=[2,2], n_dim_list=[[12,12],[12]], enable_skip={}, transform_split=0,Lagrange_multip=300000, acc_bpp=0)
vq.fit_distributed('./unit/', 2)