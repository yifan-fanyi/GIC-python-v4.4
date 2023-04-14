
from cmath import rect


def Times(a, b):  # str concat by times
    r = a
    if type(b) == list:
        for i in b:
            if len(r) == 0:
                r = i
            r += '*'+i
        return '('+r+')'
    else:       
        if len(a) == 0:
            return b
        return '('+a+'*'+b+')'

def Divide(a, b): # str concat by divide
    r = a
    if type(b) == list:
        for i in b:
            if len(r) == 0:
                r = i
            r += '/'+i
        return '('+r+')'
    else:       
        if len(a) == 0:
            return b
        return '('+a+'/'+b+')'

def Add(a, b): # str concat by plus
    r = a
    if type(b) == list:
        for i in b:
            if len(r) == 0:
                r = i
            r += '+'+i
        return '('+r+')'
    else:       
        if len(a) == 0:
            return b
        return '('+a+'+'+b+')'


def matrix_multi(win):
    win = str(win)
    # win**2*(2*win-1)
    return Times(Times(win, win), Add(Times('2', win), '(-1)'))

def Grid(n, win_list):
    recept = 1
    grid = pow(2,n)
    c = ''
    for win in win_list:
        recept *= win
        c = Add(c, Times(str((grid//recept)**2), matrix_multi(str(win))))
    return c

def Lanczos(img_size):
    img_size = str(img_size)
    return Times(str(80), [img_size, img_size])

def OneChannel(g3win_list, g5win_list, g8win_list):
    c = Add(Grid(3,g3win_list), [Grid(5, g5win_list), 
                                        Grid(8, g8win_list), 
                                        Lanczos(32), 
                                        Lanczos(256)])
    return c
def GICv4d0():
    p = OneChannel([2,2,2], [4,2,2,2], [2,2,2,2,2,2,2,2])
    q = OneChannel([4,2],[8,2,2], [8,4,2,4])
    r = OneChannel([4,2],[8,2,2], [16,4,2,2])
    return Add(p, [q,r])

if __name__ == "__main__":
    print(GICv4d0())