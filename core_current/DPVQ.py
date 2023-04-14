from core.VQ import VQ


# RD is local based, distortion use MAD 
# different from rdVQ1.py which is global MSE based RD
# no content adaptive codebook
#

class DPVQ:
    def __init__(self, n_clusters_list, dim_list, Lagrange_multip):
        self.n_clusters_list = n_clusters_list
        self.dim_list = dim_list
        self.Lagrange_multip = Lagrange_multip
        self.vq_list = []
        self.acc_bpp = 0
        self.BITSTREAM = {}

    def fit(self, X):
        for i in range(len(self.n_clusters_list)):
            vq = VQ(n_clusters=self.n_clusters_list[i], 
                    dim=self.dim_list[i], 
                    Lagrange_multip=self.Lagrange_multip).fit(X)
            self.vq_list.append(vq)
        return self
    
    def predict(self, X, acc_bpp=0):
        self.acc_bpp = acc_bpp
        # iiX = X.copy()
        for i in range(len(self.n_clusters_list)):
            iX = self.vq_list[i].predict(X)
            self.acc_bpp += self.vq_list[i].getBITSTREAM_LEN()
            self.BITSTREAM[i] = self.vq_list[i].getBITSTREAM()
            X -= iX
        return X

    def getBITSTREAM(self):
        return self.BITSTREAM
    def getACC_BPP(self):
        return self.acc_bpp
    def reduce(self):
        for i in range(len(self.n_clusters_list)):
            self.vq_list[i].reduce()
        self.BITSTREAM = {}
        self.acc_bpp = 0
        return self