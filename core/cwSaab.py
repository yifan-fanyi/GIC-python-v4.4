import numpy as np
from core.util.myPCA import myPCA
from core.util import Shrink, invShrink, load_pkl, write_pkl
from core.util import Time

class cwSaab:
    def __init__(self, win, TH=-1, channelwise=True, transform_split=0):
        self.win = win
        self.TH = TH
        self.PCA_list = []
        self.Shape = []
        self.idx = []
        self.channelwise = channelwise
        self.transform_split = transform_split

    def channel_wise_fit(self, X, win):
        if self.channelwise == True:
            pca_list = []
            for i in range(X.shape[-1]):
                tmp = Shrink(X[:,:,:,i:i+1], win)
                pca = myPCA().fit(tmp.reshape(-1, tmp.shape[-1]))
                pca_list.append(pca)
            return pca_list
        else:
            tmp = Shrink(X, win)
            pca = myPCA().fit(tmp.reshape(-1, tmp.shape[-1]))
            return pca
    
    def channel_wise_transform(self, X, win, pca_list):
        S = [X.shape[0], X.shape[1]//win, X.shape[2]//win, -1]
        if self.channelwise == True:
            tX = []
            for i in range(X.shape[-1]):
                tmp = Shrink(X[:,:,:,i:i+1], win)
                tmp = pca_list[i].transform(tmp.reshape(-1, tmp.shape[-1]))
                tX.append(tmp)
            return np.concatenate(tX, -1).reshape(S)
        else:
            tmp = Shrink(X, win)
            tmp = pca_list.transform(tmp.reshape(-1, tmp.shape[-1]))
            return tmp.reshape(S)

    def channel_wise_inverse_transform(self, tX, win, pca_list):
        S = (list)(tX.shape)
        S[-1] = -1
        tX = tX.reshape(-1, tX.shape[-1])
        if self.channelwise == True:
            X = []
            for i in range(0, tX.shape[-1], win**2):
                tmp = pca_list[i//(win**2)].inverse_transform(tX[:,i:i+win**2]).reshape(S)
                X.append(invShrink(tmp, win))
            return np.concatenate(X, -1)
        else:
            tmp = pca_list.inverse_transform(tX).reshape(S)
            return invShrink(tmp, win)

    def check_split(self, tX, islast=False):
        idx = np.zeros(tX.shape[-1], dtype=bool)
        if islast == True:
            return idx
        for i in range(tX.shape[-1]):
            #if np.mean(np.square(tX[:,:,:,i])) <= self.TH:
            if i > self.transform_split:
                idx[i] = False
            else:
                idx[i] = True
        print('%d channel is transformed to next'%np.sum(idx))
        return idx

    @Time
    def fit(self, X):
        tmp = Shrink(X, self.win[0])
        self.Shape.append([-1, tmp.shape[1], tmp.shape[2], tmp.shape[3]])
        tmp = tmp.reshape(-1, tmp.shape[-1])
        pca = myPCA().fit(tmp)
        tX = pca.transform(tmp).reshape(self.Shape[-1])
        self.PCA_list.append(pca)
        idx = self.check_split(tX)
        self.idx.append(idx)
        for i in range(1, len(self.win)):
            X = tX[:, :, :, self.idx[i-1]]
            pca_list = self.channel_wise_fit(X, self.win[i])
            self.PCA_list.append(pca_list)
            tX = self.channel_wise_transform(X, self.win[i], pca_list)
            idx = self.check_split(tX, i == len(self.win)-1)
            self.idx.append(idx)
        return self
    
    def transform(self, X):
        res = []
        tmp = Shrink(X, self.win[0])
        tmp = tmp.reshape(-1, tmp.shape[-1])
        tX = self.PCA_list[0].transform(tmp).reshape(self.Shape[0])
        res.append(tX[:,:,:,self.idx[0] == False])
        for i in range(1, len(self.win)):
            X = tX[:, :, :, self.idx[i-1]]
            tX = self.channel_wise_transform(X, self.win[i], self.PCA_list[i])
            res.append(tX[:,:,:,self.idx[i] == False])#.astype('float16'))
        return res

    def transform_distributed(self, root, n_file):
        for fileID in range(n_file):
            X = load_pkl(root+'/'+str(fileID)+'.spatial_data')
            res = self.transform(X)
            write_pkl(root+'/'+str(fileID)+'.cwsaab', res)
            write_pkl(root+'/'+str(fileID)+'.iR', res[-1])
            
    def inverse_transform(self, rX):
        tX = rX[-1]
        for i in range(len(rX)-1, 0, -1):
            tX = self.channel_wise_inverse_transform(tX,self.win[i],self.PCA_list[i])
            nX = np.zeros((tX.shape[0], tX.shape[1], tX.shape[2], len(self.idx[i-1])))
            nX[:,:,:,self.idx[i-1] == False] = rX[i-1]
            nX[:,:,:,self.idx[i-1] == True] = tX
            tX = nX
        S = tX.shape
        tX = tX.reshape(-1, tX.shape[-1])
        tX = self.PCA_list[0].inverse_transform(tX).reshape(S)
        tX = invShrink(tX, self.win[0])
        return tX

    def inverse_transform_one(self, tX, rX, i):
        if i > 0:
            tX = self.channel_wise_inverse_transform(tX,self.win[i],self.PCA_list[i])
            nX = np.zeros((tX.shape[0], tX.shape[1], tX.shape[2], len(self.idx[i-1])))
            nX[:,:,:,self.idx[i-1] == False] += rX
            nX[:,:,:,self.idx[i-1] == True] += tX
            tX = nX
        else:
            S = tX.shape
            tX = tX.reshape(-1, tX.shape[-1])
            tX = self.PCA_list[0].inverse_transform(tX).reshape(S)
            tX = invShrink(tX, self.win[0])
        return tX
    
    def inverse_transform_one_distributed(self, root, n_file, level):
        for fileID in range(n_file):
            tX = load_pkl(root+'/'+str(fileID)+'.cwsaab')
            iR = load_pkl(root+'/'+str(fileID)+'.iR')
            if level > 0:
                iR = self.inverse_transform_one(iR, tX[level-1], level)
            else:
                iR = self.inverse_transform_one(iR, None, level)
            write_pkl(root+'/'+str(fileID)+'.iR', iR)