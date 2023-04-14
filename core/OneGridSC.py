from core.util import Time, myLog
import pickle
import numpy as np
from core.util.myPCA import myPCA
from core.util.ReSample import resize
from core.util.evaluate import MSE

myLog('<FRAMEWORK> OneGrid 2022.12.09')

def pca_color(X, pca=None):
    S = X.shape
    X = X.reshape(-1, X.shape[-1])
    if pca is None:
        pca = myPCA(-1, toint=True)
        pca.fit(X)
    X = pca.transform(X)
    return X.reshape(S), pca

def pca_invcolor(X, pca=None):
    S = X.shape
    X = X.reshape(-1, X.shape[-1])
    X = pca.inverse_transform(X)
    return X.reshape(S)

def pca_color_single(X, color=None):
    pca_list = []
    tX = []
    for i in range(len(X)):
        if color is not None:
            a, _ = pca_color(X[i], color[i])
        else:
            a, b = pca_color(X[i])
            pca_list.append(b)
        tX.append(a)
    return np.array(tX), pca_list

def pca_invcolor_single(X, pca_list):
    iX = []
    for i in range(len(X)):
        iX.append(pca_invcolor(X[i], pca_list[i]))
    return np.array(iX)

def load_color(root, Y32):
    try:
        with open(root+'color_train_'+str(Y32.shape[0])+'.pkl', 'rb') as f:
            color = pickle.load(f)
    except:
        _, color = pca_color_single(Y32)
#         with open(root+'color_train_'+str(Y32.shape[0])+'.pkl', 'wb') as f:
#             pickle.dump(color, f, 4)
    return color
def remove_mean(X):
    S = X.shape
    X = X.reshape(X.shape[0],-1)
    mx = np.round(np.mean(X, axis=1, keepdims=True))
    X -= mx
    return mx, X.reshape(S)
def add_mean(mx, X):
    S = X.shape
    X = X.reshape(X.shape[0],-1)
    X += mx
    return X.reshape(S)

class OneGridSC:
    def __init__(self, grid, model_hash, model_p=None, model_q=None, model_r=None):
        self.grid = grid
        self.model_hash = model_hash
        self.model_p = model_p
        self.model_q = model_q
        self.model_r = model_r
        self.loaded = False
        with open('name.pkl', 'rb') as f:
            d = pickle.load(f)
        self.root = d['root']

    def load(self):
        try:
            with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'.model', 'rb') as f:
                d = pickle.load(f)
                self.model_p, self.model_q, self.model_r = d['P'], d['Q'], d['R']
                self.loaded = True
        except:
            try:
                with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelp.model', 'rb') as f:
                    self.model_p = pickle.load(f)['P']
                try:
                    self.model_p = self.clear(self.model_p)
                except:
                    print('caannotclear')
                with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelq.model', 'rb') as f:
                    self.model_q = pickle.load(f)['Q']
                try:
                    self.model_q = self.clear(self.model_q)
                except:
                    pass
                with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelr.model', 'rb') as f:
                    self.model_r = pickle.load(f)['R']  
                try:
                    self.model_r = self.clear(self.model_r)
                except:
                    pass
                with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'.model', 'wb') as f:
                    pickle.dump({'P':self.model_p, 'Q':self.model_q, 'R':self.model_r},f,4)
                    self.loaded = True
            except:
                print('cannot load')

    def clear(self, model):
        for k in model.Huffman.keys():
            for kk in model.Huffman[k].keys():
                if model.Huffman[k][kk] is not None:
                    try:
                        model.Huffman[k][kk].clear()
                        model.Huffman[k][kk].cent = []
                    except:
                        pass
        return model
    
    @Time
    def fit(self, iX, rX, color=None):
        myLog('---------------s-----------------')
        myLog('Grid=%d'%self.grid)
        print(rX.shape)
        self.load()
        if self.loaded == True:
            return self#.predict(iX, rX, color)
        iX = resize(iX, pow(2, self.grid))
        X = rX - iX
        
        mx, X = remove_mean(X)
        Y, _ = pca_color_single(X, color)
        try:
            with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelp.model', 'rb') as f:
                self.model_p = pickle.load(f)['P']
            #iRp = self.model_p.predict(Y[:,:,:,:1])
            self.model_p = None
        except:
            myLog("---> P fit")
            iRp = self.model_p.fit(Y[:,:,:,:1])
            #self.model_p.buffer  = {}
            iRp = []
            try:
                with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelp.model', 'wb') as f:
                    pickle.dump({'P':self.model_p},f,4)
                self.model_p = None
            except:
                pass
        try:
            with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelq.model', 'rb') as f:
                self.model_q = pickle.load(f)['Q']
            #iRq = self.model_q.predict(Y[:,:,:,1:2])
            self.model_q = None
        except:
            myLog("---> Q fit")
            iRq = self.model_q.fit(Y[:,:,:,1:2])
            self.model_q.buffer = {}
            iRq = []
            try:
                with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelq.model', 'wb') as f:
                    pickle.dump({ 'Q':self.model_q},f,4)
                self.model_q = None
            except:
                pass
        try:
            with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelr.model', 'rb') as f:
                self.model_r = pickle.load(f)['R']
            #iRr = self.model_r.predict(Y[:,:,:,2:])
            self.model_r = None
        except:
            myLog("---> R fit")
            iRr = self.model_r.fit(Y[:,:,:,2:])
            iRr=[]
            self.model_r.buffer = {}
            try:
                with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelr.model', 'wb') as f:
                    pickle.dump({ 'R':self.model_r},f,4)
                self.model_r = None
            except:
                pass
#         try:
#             with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelp.model', 'rb') as f:
#                 model_p = pickle.load(f)['P']
#             with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelq.model', 'rb') as f:
#                 model_q = pickle.load(f)['Q']
#             with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelr.model', 'rb') as f:
#                 model_r = pickle.load(f)['R']  
#             with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'.model', 'wb') as f:
#                 pickle.dump({'P':model_p, 'Q':model_q, 'R':model_r},f,4)
#             self.loaded = True
#         except:
#             pass

        #iR = np.concatenate([iRp, iRq, iRr], axis=-1)
        #iX = resize(iX,pow(2, self.grid)) + add_mean(mx, pca_invcolor_single(Y - iR, color))
        myLog('---------------e-----------------')
        return 0#iX

    @Time
    def predict(self, iX, rX, color=None, refX=None, new_lambda=None, loadonce=True, skip={}):
        myLog('---------------s-----------------')
        myLog('Grid=%d'%self.grid)
        if loadonce == True:
            self.load()
        iX = resize(iX, pow(2, self.grid))
        X = rX - iX
        mx, X = remove_mean(X)
        Y, _ = pca_color_single(X, color)
        fast = False
        if iX.shape[0] > 200:
            fast = True
        myLog("---> P predict")
        if loadonce == False:
            with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelp.model', 'rb') as f:
                self.model_p = pickle.load(f)['P']
        if new_lambda is not None:
            self.model_p.Lagrange_multip = new_lambda[0]
        iRp = self.model_p.predict(Y[:,:,:,:1], fast=fast, skip=skip.get('P',[]))
        self.model_p =  None
        myLog("---> Q predict")
        if loadonce == False:
            with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelq.model', 'rb') as f:
                self.model_q = pickle.load(f)['Q']
        if new_lambda is not None:
            self.model_q.Lagrange_multip = new_lambda[1]
        iRq = self.model_q.predict(Y[:,:,:,1:2], fast=fast, skip=skip.get('Q',[]))
        self.model_q.buffer  = None
        myLog("---> R predict")
        if loadonce == False:
            with open(self.root+'G'+str(self.grid)+'_'+self.model_hash+'_modelr.model', 'rb') as f:
                self.model_r = pickle.load(f)['R']
        if new_lambda is not None:
            self.model_r.Lagrange_multip = new_lambda[2]
        iRr = self.model_r.predict(Y[:,:,:,2:], fast=fast, skip=skip.get('R',[]))
        self.model_r  = None 
        iR = np.concatenate([iRp, iRq, iRr], axis=-1)
        myLog('<INFO> local MSE_p=%4.3f MSE_q=%4.3f MSE_r=%4.3f'%(MSE(iRp, np.zeros_like(iRp)),
                                                                  MSE(iRq, np.zeros_like(iRq)),
                                                                  MSE(iRr, np.zeros_like(iRr))))
        iX = resize(iX,pow(2, self.grid)) + add_mean(mx, pca_invcolor_single(Y - iR, color))
        myLog('<INFO> local MSE=%f'%MSE(rX, iX))
        if refX is not None:
            myLog('<MSE> global MSE=%f'%MSE(refX, resize(iX, refX.shape[1])))
        myLog('---------------e-----------------')
        return iX


                                                    





