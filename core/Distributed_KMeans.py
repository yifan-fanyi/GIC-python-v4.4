# TODO: speed up the Distributed_KMeans.write_UNCHAR
#                    Distributed_KMeans.write_SHORT
#                    Distributed_KMeans.read
# 
# A python wrap of Distributed KMeans from Dr. Ioannis
#  rmse checked after each iteration, if rmse change < max_err or reach the max_iter, the kmeans will stop
#  then the current codebook will be loaded to a myKMeans object for easy python usage.
#  For UNCHAR, the loading works good and have been test several times to make sure the result is the same as C out put
#  For SHORT, I only test the logic, (48 dimension vector, so that stride is 64 and all the last 16 dim of the load codebook is 0)
# 
# The Distributed_KMeans.map_stage and Distributed_KMeans.reduce_stage will run the map and reduce of the C code by calling the pre-compiled file
# Distributed_KMeans.gen_list will generate the map_state.list (only happens at first iteration)
# 
# Requirement: the data can be stored in multiple files, 
#       but each file should contain same amount of frames
#       each frame must have the same resolution
#       the file saving can use Distributed_KMeans.write_UNCHAR or Distributed_KMeans.write_SHORT 
#       loading the saved file can call Distributed_KMeans.read
#   designed for 444
#   Distributed_KMeans.fit has ben tested for both UCHAR and SHORT
#   Distributed_KMeans.predict and Distributed_KMeans.inverse_predict haven;t been tested
import numpy as np
import os
from core.util.myKMeans import *
import subprocess
from core.util import Shrink, invShrink
import pickle
from core.util.evaluate import MSE 
import copy

def write_UNCHAR(X, file, offset=128, truewrite=True):
    X += offset
    X[X<0] = 0
    X[X>255] = 255
    print('expected size %d bytes'%(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3]))
    if truewrite == True:
        Xt = copy.deepcopy(X)
        Xt = Xt.astype(np.int8).transpose((0, 3, 1, 2)).reshape(-1)
        with open(file+'.data', 'wb') as f:
            f.write(Xt)
    return X

def write_SHORT(X, file, truewrite=True):
    print('expected size %d bytes'%(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3]))
    if truewrite == True:
        Xt = copy.deepcopy(X)
        Xt = Xt.astype(np.int16).transpose((0, 3, 1, 2)).reshape(-1)
        with open(file+'.data', 'wb') as f:
            f.write(Xt)
       
    return X

def read(file, datatype="SHORT", size=256, frame_each_file=1000):
    pix_num = size * size * 3
    X = np.zeros(frame_each_file * pix_num)
    dtype = np.int8
    if datatype == 'SHORT':
        dtype = np.int16
    with open(file, 'rb') as f:
        for k in range(frame_each_file):
            X[k*pix_num:(k+1)*pix_num] = np.fromfile(f, dtype=dtype, count=pix_num, sep='')
    X = X.reshape((frame_each_file, 3, size, size)).transpose((0,2,3,1))
    return X.astype('float32')
   
class Distributed_KMeans:
    def __init__(self, n_clusters, size, win, datatype, frame_each_file, n_frames, max_iter=10000, max_err=1e-7):
        self.folder = ''
        self.file_list = ''
        self.n_clusters = n_clusters
        self.size = size
        self.win = win
        self.datatype = datatype
        self.frame_each_file = frame_each_file
        self.n_frames=n_frames
        self.last_rmse = -1
        self.max_iter = max_iter
        self.max_err = max_err
        self.KM = None
        self.fast = True

    def write_UNCHAR(self, X, file, offset=128, truewrite=True):
        X += offset
        X[X<0] = 0
        X[X>255] = 255
        print(np.mean(X), np.min(X), np.max(X))
        print('expected size %d bytes'%(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3]))
        if truewrite == True:
            if self.fast == True:
                Xt = copy.deepcopy(X)
                Xt = Xt.astype(np.int8).transpose((0, 3, 1, 2)).reshape(-1)
                with open(file+'.data', 'wb') as f:
                    f.write(Xt)
            else:
                with open(file+'.data', 'wb') as f:
                    for k in range(X.shape[0]):
                        for c in range(X.shape[3]):
                            for i in range(X.shape[1]):
                                for j in range(X.shape[2]):
                                    a = int(X[k,i,j,c])
                                    f.write(a.to_bytes(1, 'little'))
        return X

    def write_SHORT(self, X, file, truewrite=True):
        print(np.mean(X), np.min(X), np.max(X))
        print('expected size %d bytes'%(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3]))
        if truewrite == True:
            if self.fast == True:
                Xt = copy.deepcopy(X)
                Xt = Xt.astype(np.int16).transpose((0, 3, 1, 2)).reshape(-1)
                with open(file+'.data', 'wb') as f:
                    f.write(Xt)
            else:
                with open(file+'.data', 'wb') as f:
                    for k in range(X.shape[0]):
                        for c in range(X.shape[3]):
                            for i in range(X.shape[1]):
                                for j in range(X.shape[2]):
                                    a = int(X[k,i,j,c])
                                    f.write(a.to_bytes(2, 'little', signed=True))
        return X
    
    def read(self, fileID):
        if self.fast == True:
            pix_num = self.size * self.size * 3
            X = np.zeros(self.frame_each_file * pix_num)
            dtype = np.int8
            if self.datatype == 'SHORT':
                dtype = np.int16
            with open(self.folder+'/'+self.file_list[fileID], 'rb') as f:
                for k in range(self.frame_each_file):
                    X[k*pix_num:(k+1)*pix_num] = np.fromfile(f, dtype=dtype, count=pix_num, sep='')
            X = X.reshape((self.frame_each_file, 3, self.size, self.size)).transpose((0,2,3,1))
            return X.astype('float32')
        else:
            X = np.zeros((self.frame_each_file, self.size, self.size, 3))
            with open(self.folder+'/'+self.file_list[fileID], 'rb') as f:
                for k in range(X.shape[0]):
                    for c in range(X.shape[3]):
                        for i in range(X.shape[1]):
                            for j in range(X.shape[2]):
                                if self.datatype == "SHORT":
                                    X[k,i,j,c] = int.from_bytes(f.read(2), 'little', signed=True)
                                else:
                                    X[k,i,j,c] = int.from_bytes(f.read(1), 'little')
            return X.astype('float32')
    
    def read_codebook(self, file):
        dim = self.win**2 * 3
        nc = self.n_clusters
        r = 1
        if self.datatype == 'SHORT':
            r = 2
        stride = 32*((dim*r+31)//32)//r
        if self.datatype == "UCHAR":
            offset = (8+1+1) * stride + 4+8+4 
        else:
            offset = (8+2+2) * stride + 4+8+4 
        cent_size = (stride) * nc
        # print(stride, offset)
        with open(file, 'rb') as f:
            _ = f.read(offset)
            cent = []
            for i in range(cent_size):
                if self.datatype == "SHORT":
                    a = int.from_bytes(f.read(2), "little", signed=True)
                else:
                    a = int.from_bytes(f.read(1), "little")
                cent.append(a)
        cent = np.array(cent).reshape(nc, -1).astype('float32')
        assert np.sum(np.abs(cent[:, dim:])) == 0, 'Error offset '+str(np.sum(np.abs(cent[:, dim:])))
        return cent[:,:dim]
    
    def map_stage(self, fileID, start_frame, n_frames=4000):
        # cmd = './distributed-kmeans-main/map_reduce_kmeans ' + self.datatype
        # cmd += ' ' + self.folder + '/' + self.file_list[fileID]
        # cmd += ' ' + str(self.size) + ' ' + str(self.size)
        # cmd += ' ' + str(self.win) + ' ' + str(self.win)
        # cmd += ' ' + str(self.n_clusters)
        # cmd += ' ' + str(start_frame) + ' ' + str(n_frames)
        # cmd += ' ' + self.folder + '/train.state'
        # # cmd += ' >> log.txt'
        # os.system(cmd)

        cmd = ['./distributed-kmeans-main/map_reduce_kmeans',
               self.datatype,
               self.folder + '/' + self.file_list[fileID],
               str(self.size),str(self.size),
               str(self.win),str(self.win),
               str(self.n_clusters),
               str(start_frame),str(n_frames),
               self.folder + '/train.state']
        try:
            return subprocess.check_output(cmd)
        except subprocess.CalledProcessError as err:
            print(err)
            exit()
        
    def gen_list(self, fileID, start_frame, n_frames=4000):
        os.system('rm -f '+self.folder +'/map_state.list')
        cmd = 'echo ' + self.folder + '/'
        cmd += self.file_list[fileID] + '_' 
        cmd += str(self.size) + '_' + str(self.size) + '_'
        cmd += str(self.win) + '_' + str(self.win) + '_'
        cmd += str(self.n_clusters) +'_' 
        cmd += str(start_frame) + '_' + str(n_frames)
        cmd += '_kmeans.state >> '+ self.folder +'/map_state.list'
        os.system(cmd)
        
    def reduce_stage(self):
        # cmd = './distributed-kmeans-main/map_reduce_kmeans '
        # cmd += self.datatype 
        # cmd += ' ' + self.folder + '/map_state.list'
        # cmd += ' ' + self.folder + '/train.state'
        # # cmd += ' >> log.txt'
        # os.system(cmd)

        cmd = ['./distributed-kmeans-main/map_reduce_kmeans', 
               self.datatype,
               self.folder + '/map_state.list',
               self.folder + '/train.state']
        try:
            return str(subprocess.check_output(cmd))
        except subprocess.CalledProcessError as err:
            print(err)
            exit()
        
    def early_stop(self, msg):
        msg = msg.split(" ")
        for i in msg:
            if i[:5] == 'RMSE=':
                j = i.split('=')
                rmse = float(j[1])
                if abs(rmse - self.last_rmse) < self.max_err:
                    return True
                else:
                    self.last_rmse = rmse
                    return False
                
    def fit(self, folder, file_list):
        self.folder, self.file_list = folder, file_list
        for it in range(self.max_iter):
            for fileID in range(len(self.file_list)):
                for start_frame in range(0, self.frame_each_file, self.n_frames):
                    if it == 0:
                        self.gen_list(fileID, start_frame, self.n_frames)
                    self.map_stage(fileID, start_frame, self.n_frames)
            msg = self.reduce_stage()
            if self.early_stop(msg) == True:
                print('Early terminate @iter=%d'%it)
                break
            print(msg)
        cent = self.read_codebook(self.folder+'/train.state.codebook')
        self.KM = myKMeans(-1).fit(X=None, cluster_centers=cent)
        return self
    
    def predict(self, folder, file_list, residual_folder, th):
        self.folder, self.file_list = folder, file_list
        for fileID in range(len(self.file_list)):
            X = self.read(fileID)
            X = Shrink(X, self.win)
            label = self.KM.predict(X)
            iX = self.KM.inverse_predict(label)
            S = (X.shape[0], X.shape[1],X.shape[2],-1)
            X, iX = X.reshape(-1, X.shape[-1]), iX.reshape(-1, iX.shape[-1])
            mse = (np.mean(np.square((X-iX).astype('float32')),axis=1))
            omse =  np.mean(np.square(X.astype('float32')), axis=1)
            dmse = omse-mse
            idx = dmse > th
            iX[idx==False] *=0
            X, iX = X.reshape(S), iX.reshape(S)
            X, iX = invShrink(X, self.win), invShrink(iX, self.win)
            if self.datatype == "SHORT":
                self.write_SHORT(X-iX, 
                                    residual_folder+'/'+self.file_list[fileID],
                                    True)
            else:
                self.write_UNCHAR(X-iX,
                                    residual_folder+'/'+self.file_list[fileID])
        return 
    
    def inverse_predict(self, folder, file_list, residual_folder=None):
        self.folder, self.file_list = folder, file_list
        for fileID in range(len(self.file_list)):
            X = self.read(fileID)
            with open(self.folder+'/'+self.file_list[fileID]+'.label', 'rb') as f:
                label = pickle.load()
            iX = self.KM.inverse_predict(label)
            iX = invShrink(iX, self.win)
            print('<INFO> file=%s, MSE=%f'%(self.folder+'/'+self.file_list[fileID], MSE(X,iX)))
            if residual_folder is not None: 
                if self.datatype == "SHORT":
                    self.write_SHORT(X-iX, 
                                     residual_folder+'/'+self.file_list[fileID],
                                     True)
                else:
                    self.write_UNCHAR(X-iX,
                                      residual_folder+'/'+self.file_list[fileID])
        return 

if __name__ == "__main__":
    

    dkm = Distributed_KMeans(n_clusters=16, 
                            size=256, 
                            win=4, 
                            datatype="SHORT", 
                            frame_each_file=200, 
                            n_frames=200).fit(folder='.', 
                                              file_list=['test_short1.data', 'test_short2.data'], )
    