# CABAC for VQ indices
import numpy as np
from core.util.Arithmetic import Arithmetic
import warnings
warnings.filterwarnings("ignore")
from core.util.myKMeans import myKMeans
from core.util import Shrink
from core.util.Huffman import Huffman
class BAC:
    def __init__(self, A=0x10000, C=0, state=12, MPS=0):
        self.A = A
        self.C = C
        self.state = state
        self.carry = False
        self.MPS = MPS
        self.StateTable = [[0, 0x59EB, 1, -1],[1, 0x5522, 1, 1],[2, 0x504F, 1, 1],[3, 0x4B85, 1, 1],
                            [4, 0x4639, 1, 1],[5, 0x415E, 1, 1],[6, 0x3C3D, 1, 1],[7, 0x375E, 1, 1],
                            [8, 0x32B4, 1, 2],[9, 0x2E17, 1, 1],[10, 0x299A, 1, 2],[11, 0x2516, 1, 1],
                            [12, 0x1EDF, 1, 1],[13, 0x1AA9, 1, 1],[14, 0x174E, 1, 1],[15, 0x1424, 1, 1],
                            [16, 0x119C, 1, 1],[17, 0x0F6B, 1, 2],[18, 0x0D51, 1, 2],[19, 0x0BB6, 1, 1],
                            [20, 0x0A40, 1, 2],[21, 0x0861, 1, 2],[22, 0x0706, 1, 2],[23, 0x05CD, 1, 2],
                            [24, 0x04DE, 1, 1],[25, 0x040F, 1, 2],[26, 0x0363, 1, 2],[27, 0x02D4, 1, 2],
                            [28, 0x025C, 1, 2],[29, 0x01F8, 1, 2],[30, 0x01A4, 1, 2],[31, 0x0160, 1, 2],
                            [32, 0x0125, 1, 2],[33, 0x00F6, 1, 2],[34, 0x00CB, 1, 2],[35, 0x00AB, 1, 1],
                            [36, 0x008F, 1, 2],[37, 0x0068, 1, 2],[38, 0x004E, 1, 2],[39, 0x003B, 1, 2],
                            [40, 0x002C, 1, 2],[41, 0x001A, 1, 3],[42, 0x000D, 1, 2],[43, 0x0006, 1, 2],
                            [44, 0x0003, 1, 2],[45, 0x0001, 0, 1]]
        self.output = ''
        self.Qe = self.StateTable[state][1]

    def toInt16(self, C):
        while C >= 65536:
            C -= 65536
        return C

    def Renormalize(self):
        if self.carry == True:
            self.output += "1"
            self.carry = False
        while self.A < 0x8000:
            self.A = self.A * 2
        if self.C & (0x8000) == 0:
            self.output += "0"
        else:
            self.output += "1"
        self.C = self.C * 2
        self.C = self.toInt16(self.C);


    def encode(self, st):
        for i in range(len(st)): 
            if st[i] == self.MPS: 
                self.A -= self.Qe
                if self.A < 0x8000:
                    if self.A < self.Qe:
                        if self.A+self.C > 0xffff:
                            self.carry = True
                        self.C += self.A
                        self.C = self.toInt16(self.C)
                        self.A = self.Qe;
                    self.state += self.StateTable[self.state][2]
                    self.Qe = self.StateTable[self.state][1]
                    self.Renormalize()
            else:
                self.A -= self.Qe
                if self.A >= self.Qe:
                    if self.A + self.C > 0xffff:
                        self.carry = True
                    self.C += self.A
                    self.C = self.toInt16(self.C)
                    self.A = self.Qe
                self.state -= self.StateTable[self.state][3]
                if self.StateTable[self.state][3] < 0:
                    self.MPS = 1 - self.MPS
                self.Qe = self.StateTable[self.state][1]
                self.Renormalize()
        return self

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
    def __init__(self, nc, cent):
        self.CModel = {}
        self.map = {}
        self.nc = nc
        self.cent = cent
        self.binary = False
        self.d = {}
        self.Huffman = Huffman()

    def mappingFunc(self, val):
        try: 
            return self.map[val]
        except:
            print(val,self.map)
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

    def get_context(self, label, idx, k, i, j):
        return str(self.get_val(label, idx, k, i-1, j-1)) + \
                str(self.get_val(label, idx, k, i-1, j)) + \
                str(self.get_val(label, idx, k, i-1, j+1)) + \
                str(self.get_val(label, idx, k, i, j-1))

    def fit(self, label, idx, group=3):
        km = myKMeans(group).fit(self.cent)
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
                        context = self.get_context(label, idx, k, i, j)
                        if context not in self.d.keys():
                            self.d[context] = []
                        self.d[context].append(label[k,i,j,0])
            for k in self.d.keys():
                #self.CModel[k] = Arithmetic(mode='fix', n_symbols=self.nc, num_state_bits=32).fit(self.d[k]+np.arange(self.nc).tolist())
                self.CModel[k] = Huffman().fit(np.array(self.d[k]+np.arange(self.nc).tolist()))
        try:
            self.Huffman.fit(np.array(label.reshape(-1)[idx.reshape(-1)].tolist()+np.arange(self.nc).tolist()))
        except:
            self.Huffman=None
        self.clear()
        return self

    def encode(self, label, idx):
        if label.shape[0] > 200:
            p = int(0.5 * np.sum(idx))
            if p < 1:
                return ''
            if self.Huffman is None:
                return ''
            return self.Huffman.encode(label.reshape(-1)[idx.reshape(-1)])[p:]
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
