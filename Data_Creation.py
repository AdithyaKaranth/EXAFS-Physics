
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:55:44 2019
@author: Adithya Karantha
"""

import larch
import random
from larch_plugins.xafs import feffdat
from larch import Interpreter
import operator 
import numpy as np
from operator import itemgetter
from larch_plugins.io import read_ascii
from larch_plugins.xafs import autobk
import datetime
import time
import random

mylarch = Interpreter()

#range for rdm num generator
S02 = (np.linspace(5,15,146) * 0.01).tolist()
E0 = (np.linspace(-500,500,1001) * 0.01).tolist()
SIGMA2 = (np.linspace(1,15,16) * 0.001).tolist()
DELTAR = (np.linspace(-20,20,41) * 0.01).tolist()

#len(rangeA)
#print(len(rangeA))
#print(len(rangeB))
#print(len(rangeC))
#print(len(rangeD))
#rangeB.append(0)
#rangeC.append(0)
#rangeD.append(0)
print(min(S02)," + ",max(S02))
print(min(E0)," + ",max(E0))
print(min(SIGMA2)," + ",max(SIGMA2))
print(min(DELTAR)," + ",max(DELTAR))
#front = '/Users/42413/Documents/GitHub/EXAFS/Cu Data/path Data/feff'
front = '/Users/user/EXAFS-Physics/Cu Data/path Data/feff'
end = '.dat'

random.seed(19)
def generate_y():
    x_total = [0]*(281)
    y = []
    #print(y_total)
    for i in range(1,11):
        if i < 10:
            filename = front+'000'+str(i)+end
        elif i< 100:
            filename = front+'00'+str(i)+end
        else:
            filename = front+'0'+str(i)+end
        s02 = random.choice(S02)
        e0 = 1.1 ## Domain knowledge
        sigma2 = random.choice(SIGMA2)
        deltar = random.choice(DELTAR)
        
        #print(filename)
        #print(a,b,c,d)
        path=feffdat.feffpath(filename, s02= str(s02), e0= str(e0), sigma2= str(sigma2), deltar= str(deltar), _larch=mylarch)
        y.extend([s02,e0,sigma2,deltar])
        #print(y)
        feffdat.path2chi(path, _larch=mylarch)
        x = path.chi
        x = x[60:341]
        #print("-------------------------------------------------------")
        #print(y)
        #print("------------------------------------------------------")
        for k in range(len(x)):
           x_total[k] += x[k]
    return(x_total, y)

#print(len(y))
#print(min(x), max(x))

x = []
y = []

for i in range(0,10000):
    X,Y = generate_y()
    x.append(X)
    y.append(Y)
    
#y.append([0.9,1.1,0.004,0.03])

#filename = '/Users/user/EXAFS-Physics/Cu Data/path Data/feff0001.dat'
#path=feffdat.feffpath(filename, s02= str(y[10000][0]), e0= str(y[10000][1]), sigma2= str(y[10000][2]), deltar= str(y[10000][3]), _larch=mylarch)
#print(y)
#feffdat._path2chi(path, _larch=mylarch)
#X = path.chi
#X = X[60:341]
#x.append(X)
#a = random.choice(rangeA)
#b = random.choice(rangeB)
#c = random.choice(rangeC)
#d = random.choice(rangeD) 
"""
## Check
k_value = (np.linspace(0,2000,401)*0.01).tolist()
k_value = np.asarray(k_value)
<<<<<<< HEAD
chi = y
=======
k_value = k_value[60:341]
chi = np.asarray(x[4996])
>>>>>>> d5a59f1c6600b79b50bd3ee5553a53c3a60964e1
import matplotlib.pyplot as plt
plt.plot(k_value,chi*k_value**2)
plt.show()
"""

"""
path=feffdat.feffpath('/Users/user/EXAFS-master/Cu Data/path Data/feff0001.dat', s02= str(rangeA[20]), e0= str(rangeB[20]), sigma2= rangeD[20], deltar= rangeC[20], _larch=mylarch)
feffdat._path2chi(path, _larch=mylarch)
y = path.chi
#print(y)
"""

import numpy as np
x = np.asarray(x)
y = np.asarray(y)

import pickle

with open('input.pickle', 'wb') as f:
    pickle.dump(x, f,)

with open('output.pickle', 'wb') as f:
    pickle.dump(y,f)

