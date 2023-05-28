# CABAC for VQ indices
import numpy as np
from core.util.Arithmetic import Arithmetic
import warnings
warnings.filterwarnings("ignore")
from core.util.myKMeans import myKMeans
from core.util import Shrink
from core.util.Huffman import Huffman
from core.util.mydKMeans import load_pkl, write_pkl
# VQentropy v2023.03.26

def entropy(x, nbin,v=True):
    p = np.zeros((nbin))
    x = x.reshape(-1)
    for i in range(len(x)):
        p[x[i]] +=1.
    p = p/np.sum(p)
    if v==True:
        pass#print(p)
    return -np.sum(p * np.log2(p+1e-10))

class VQEntropy:
    def __init__(self, nc, cent, group=31):
        self.CModel = {}
        self.map = {}
        self.nc = nc
        self.cent = cent
        self.binary = False
        self.d = {}
        self.Huffman = Huffman()
        self.group=group
    def mappingFunc(self, val):
        try: 
            return self.map[val]
        except:
            return -1

    def get_val(self, label, idx, k, i, j):
        assert label.shape == idx.shape, "Shape not the same."+str(label.shape)+str(idx.shape)
        if i < 0 or i >= idx.shape[1]:
            return -1
        if j < 0 or j >= idx.shape[2]:
            return -1
        if idx[k, i, j, 0] == False:
            return -1
        return self.mappingFunc(label[k, i, j, 0])

    def get_context(self, label, idx, k, i, j, plabel=None):  
        a=''
#         a += str(self.get_val(label, idx, k, i-1, j-1)) + '~' 
        a+=        str(self.get_val(label, idx, k, i-1, j)) + '~' 
#         a+=        str(self.get_val(label, idx, k, i-1, j+1)) + '~'
        a+=        str(self.get_val(label, idx, k, i, j-1))
        return a
    
    def fit(self, label, idx, group=3, keep_fit=False, done=False, plabel=None):
        if keep_fit == False:
            km = myKMeans(self.group).fit(self.cent)
            l = km.predict(self.cent).reshape(-1)
            for i in range(len(l)):
                self.map[i] = l[i]
            self.cent = [] 
        if self.binary == False:
            for k in range(idx.shape[0]):
                for i in range(idx.shape[1]):
                    for j in range(idx.shape[2]):
                        if idx[k,i,j,0] == False:
                            continue
                        context = self.get_context(label, idx, k, i, j, plabel)
                        if context not in self.d.keys():
                            self.d[context] = np.ones(self.nc)
                        self.d[context][label[k,i,j,0]] += 1
            if done == True:
                for k in self.d.keys():
                    #self.CModel[k] = Arithmetic(mode='fix', n_symbols=self.nc, num_state_bits=32).fit(self.d[k]+np.arange(self.nc).tolist())
                    self.CModel[k] = Huffman().fit(None, hist=self.d[k])
        if done == True:
            hist = np.zeros(self.nc)
            for k in self.d.keys():
                hist += self.d[k]
            self.Huffman.fit(None, hist=hist)
            self.clear()
        return self

    def fit_distributed(self, root, n_file):
        # need to have *.label and *.dmse under root
        for fileID in range(n_file):
            label = load_pkl(root+'/'+str(fileID) +'.label')
            dmse = load_pkl(root+'/'+str(fileID) +'.dmse')
            for skip_TH in range(0, 5001, 10):
                if skip_TH == 0 and fileID == 0:
                    self.fit(label, dmse > skip_TH, group=3, keep_fit=False, done=False, plabel=None)
                elif skip_TH == 5000 and fileID == n_file-1:
                    self.fit(label, dmse > skip_TH, group=3, keep_fit=True, done=True, plabel=None)
                else:
                    self.fit(label, dmse > skip_TH, group=3, keep_fit=True, done=False, plabel=None)


    def encode(self, label, idx, fast=False):
        if fast == True:
            p = int(0.5 * np.sum(idx))
            if p < 1:
                return ''
            if self.Huffman is None:
                return ''
            return self.Huffman.encode(label.reshape(-1)[idx.reshape(-1)])
        self.clear()
        for k in range(idx.shape[0]):
            for i in range(idx.shape[1]):
                for j in range(idx.shape[2]):
                    if idx[k,i,j,0] == False:
                        continue
                    context = self.get_context(label, idx, k, i, j)
                    if context not in self.CModel.keys():
                        self.CModel[context] = Huffman().fit(np.arange(self.nc))#Arithmetic(mode='adp', n_symbols=self.nc, num_state_bits=32)
                    if context not in self.d.keys():
                        self.d[context] = []
                    self.d[context].append(label[k,i,j,0])
        st = ''

        for k in self.d.keys():
            st += self.CModel[k].encode(np.array(self.d[k]))
#         s, e = 0, 0
#         for k in self.d.keys():
#             a = np.array(self.d[k]).reshape(-1)
#             e += entropy(a,self.nc, 0) * len(a)
#             s += len(a)
        #print(e/s)
        self.clear()
        return st

    def clear(self):
        self.d = {}
        


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    def unit_VQEntropy(rX, rXt):
        print('-----------------------------------------')
        def select(X, iX, th):
            S = [X.shape[0], X.shape[1], X.shape[2], -1]
            X, iX = X.reshape(-1, X.shape[-1]), iX.reshape(-1, iX.shape[-1])
            mse = np.mean(np.square(X), axis=1)
            nmse = np.mean(np.square(X-iX), axis=1)
            dmse=  mse - nmse
            #plt.hist(dmse, bins=32)
            #plt.show()
            idx = dmse > th
            return idx.reshape(S)
        for win in [4, 8]:
            for nc in [256, 1024]:
                X, Xt = Shrink(rX.copy(), win), Shrink(rXt.copy(), win)
                km = myKMeans(nc).fit(X)
                label, labelt = km.predict(X), km.predict(Xt)
                #print('<raw entropy>',entropy(label.reshape(-1),nc), entropy(labelt.reshape(-1), nc))
                iX, iXt = km.inverse_predict(label), km.inverse_predict(labelt)
                for th in range(0, 200, 5):
                    idx, idxt = select(X.copy(), iX.copy(), th), select(Xt.copy(), iXt.copy(), th)
                    if np.sum(idx) < 1 or np.sum(idxt) < 1:
                        break
                    h = Huffman().fit(label.reshape(-1)[idx.reshape(-1)])
                    sth, stth = h.encode(label.reshape(-1)[idx.reshape(-1)]), h.encode(labelt.reshape(-1)[idxt.reshape(-1)])
                    ac = Arithmetic(mode='fix', n_symbols=nc, num_state_bits=32).fit(label.reshape(-1)[idx.reshape(-1)].tolist() + np.arange(nc).tolist())
                    stac, sttac = ac.encode(label.reshape(-1)[idx.reshape(-1)]), ac.encode(labelt.reshape(-1)[idxt.reshape(-1)])
                    print('  ---','win',win, 'nc',nc, 'th',th, np.sum(idx), np.sum(idxt), )
                    print('      Entropy',entropy(label.reshape(-1)[idx.reshape(-1)],nc), entropy(labelt.reshape(-1)[idxt.reshape(-1)],nc))
                    print('      Huffman',len(sth)/np.sum(idx), len(stth)/np.sum(idxt))
                    print('      AC',len(stac)/np.sum(idx), len(sttac)/np.sum(idxt))
                    vqe = VQEntropy(nc, km.inverse_predict(np.arange(nc).reshape(-1, 1))).fit(label, idx)
                    st = vqe.encode(label, idx)
                    # vqe = VQEntropy(nc, h, km.inverse_predict(np.arange(nc).reshape(-1, 1))).fit(label)
                    stt = vqe.encode(labelt, idxt)
                    #vqe = VQEntropy(nc, h).fit(label)
                    stt = vqe.encode(labelt, idxt)
                    print('      New', len(st)/np.sum(idx), len(stt)/np.sum(idxt))
            #         break
            #     break
            # break
        print('-----------------------------------------')

    import cv2
    from core.util.ReSample import resize
    x = []
    xt = []
    for i in range(500):
        a = cv2.imread('/Users/alex/Desktop/proj/data/train512/'+str(i)+'.png')
        x.append(a)
        if i < 186:
            b = cv2.imread('/Users/alex/Desktop/proj/data/test512/'+str(i)+'.png')
            xt.append(b)
    x, xt = np.array(x).astype('float32'), np.array(xt).astype('float32')


    unit_VQEntropy(x-resize(resize(x,256),512), xt[:]-resize(resize(xt[:],256),512))
