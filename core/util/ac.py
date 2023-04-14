

from symbol import pass_stmt


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

class HierarchyCABAC:
    def __init__(self):
        self.CModel = {}
    def get_val(self, idx, k, i, j):
        if i < 0 or i >= idx.shape[1]:
            return 0
        if j < 0 or j >= idx.shape[2]:
            return 0
        return idx[k, i, j, 0]

    def get_context(self, pidx, idx, k, i, j):
        if pidx is None:
            return str(self.get_val(idx, k, i-1, j-1)) + \
                str(self.get_val(idx, k, i-1, j)) + \
                str(self.get_val(idx, k, i-1, j+1)) + \
                str(self.get_val(idx, k, i, j-1))
        return str(pidx[k,i//self.r,j//self.r,0])+'-'+ \
                str(self.get_val(idx, k, i-1, j-1)) + \
                str(self.get_val(idx, k, i-1, j)) + \
                str(self.get_val(idx, k, i-1, j+1)) + \
                str(self.get_val(idx, k, i, j-1))
    
    def get_val1(self, idx, k, i, j):
        if i < 0:
            if j < 0:
                return -1
            if j >= idx.shape[2]:
                return -2
            return -3
        if i >= idx.shape[1]:
            if j < 0:
                return -4
            if j >= idx.shape[2]:
                return -5
            return -6
        if j < 0:
            return -7
        if j >= idx.shape[2]:
            return -8
        return idx[k, i, j, 0]
 
    def get_context1(self, pidx, idx, k, i, j):
        if pidx is None:
            return str(self.get_val1(idx, k, i-1, j-1)) +'~'+ \
               str(self.get_val1(idx, k, i-1, j)) +'~'+ \
               str(self.get_val1(idx, k, i-1, j+1)) +'~'+ \
               str(self.get_val1(idx, k, i, j-1))
        return str(pidx[k,i//self.r,j//self.r,0])+'+'+ \
               str(self.get_val1(idx, k, i-1, j-1)) +'~'+ \
               str(self.get_val1(idx, k, i-1, j)) +'~'+ \
               str(self.get_val1(idx, k, i-1, j+1))+'~' + \
               str(self.get_val1(idx, k, i, j-1))

    def encode(self, pidx, idx, mode=0):
        if pidx is not None:
            self.r = idx.shape[1]//pidx.shape[1]
        for k in range(idx.shape[0]):
            for i in range(idx.shape[1]):
                for j in range(idx.shape[2]):
                    if mode == 0:
                        context = self.get_context(pidx, idx, k, i, j)
                    else:
                        context = self.get_context1(pidx, idx, k, i, j)
                    if context not in self.CModel.keys():
                        self.CModel[context] = BAC()
                    self.CModel[context].encode([idx[k, i, j, 0]])
        st = ''
        for k in self.CModel.keys():
            st += self.CModel[k].output
        return st

if __name__ == "__main__":
    st = [1, 0, 1,1, 0,1,1]
    bac = BAC().encode(st)
    print(bac.output)