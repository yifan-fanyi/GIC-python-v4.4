# 2021.11.05
__author__ = "yifan"
__version__ = "2.5.0"

from textwrap import fill
import numpy as np
import time
import inspect
from time import gmtime, strftime, localtime
from skimage.util import view_as_windows

def Shrink(X, win):
    X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def invShrink(X, win):
    S = X.shape
    X = X.reshape(S[0], S[1], S[2], -1, 1, win, win, 1)
    X = np.moveaxis(X, 5, 2)
    X = np.moveaxis(X, 6, 4)
    return X.reshape(S[0], win*S[1], win*S[2], -1)

class OverLap:
    def __init__(self, block, boundary, pad='reflect'):
        self.block = block
        self.boundary = boundary
        self.pad = pad
        
    def padding(self, feature):
        if self.pad == 'reflect':
            feature = np.pad(feature, ((0,0),(self.boundary,self.boundary),(self.boundary,self.boundary),(0,0)), 'reflect')
        elif self.pad == 'zeros':
            feature = np.pad(feature, ((0,0),(self.boundary,self.boundary),(self.boundary,self.boundary),(0,0)), 'constant', constant_values=0)
        else:
            feature = np.pad(feature, ((0,0),(self.boundary,self.boundary),(self.boundary,self.boundary),(0,0)), 'symmetric')
        return feature
    
    def transform(self, X):
        S = (X.shape[0], X.shape[1]//self.block, X.shape[2]//self.block, -1)
        X = self.padding(X)
        res = []
        for k in range(X.shape[0]):
            for i in range(self.boundary, X.shape[1]-self.boundary, self.block):
                for j in range(self.boundary, X.shape[2]-self.boundary, self.block):
                    res.append(X[k, 
                                 i-self.boundary:i+self.boundary+self.block, 
                                 j-self.boundary:j+self.boundary+self.block])
        return np.array(res).reshape(S)
    
    def inverse_transform(self, X):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 2*self.boundary+self.block, 2*self.boundary+self.block, -1)
        res = []
        for k in range(X.shape[0]):
            for i in range(X.shape[1]):
                for j in range(X.shape[2]):
                    res.append(X[k,i,j,self.boundary:-self.boundary,self.boundary:-self.boundary])
        res = np.array(res).reshape(X.shape[0], X.shape[1], X.shape[2], self.block, self.block, -1)
        res = np.moveaxis(res, -1, 3).reshape(X.shape[0], X.shape[1], X.shape[2], -1)
        return invShrink(res, win=self.block)

def getVal(mydict, key, fillval=False):
    if key in mydict.keys():
        return mydict[key]
    if key not in ["n_threads", "verbose", "max_win", "n_levels", "n_subgroups_per_split t"]:
        print("   <Warning> Key=\"%s\" not exist, use \"%s\" as default!"%(str(key), str(fillval)))
    myLog("   <Warning> Key=\"%s\" not exist, use \"%s\" as default!"%(str(key), str(fillval)))
    return fillval

def Time(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        myLog("   <RunTime> %s: %4.1f s"%(method.__name__, (te - ts)))
        return result
    return timed

def print_name_and_args():
    caller = inspect.stack()[1][0]
    args, _, _, values = inspect.getargvalues(caller)
    st = '      '
    for i in args:
        st += i+'='+str(values[i])+'\n      '
    myLog('   args:\n', st)

def Seperate(method):
    def seperated(*args, **kw):
        myLog('################## START %s ##################'%method.__name__)
        print('################## START %s ##################'%method.__name__)
        result = method(*args, **kw)
        myLog('################## END   %s ##################'%method.__name__)
        print('################## END   %s ##################'%method.__name__)
        return result
    return seperated

def myLog(info, importance=0, verbose=3):
    print(info)
    if importance <= verbose:
        f = open("./log/log.txt", "a")
        f.write(strftime("[%Y-%m-%d %H:%M:%S]", gmtime()))
        f.write(info)
        f.write('\n')
        f.close()
