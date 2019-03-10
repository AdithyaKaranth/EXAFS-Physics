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
rangeA = (np.linspace(5,150,146) * 0.01).tolist()
rangeB = (np.linspace(-500,500,1001) * 0.01).tolist()
rangeC = (np.linspace(-20,20,41) * 0.01).tolist()
rangeD = (np.linspace(1,35,35) * 0.001).tolist()
rangeA.append(0)
#len(rangeA)
#print(len(rangeA))
#print(len(rangeB))
#print(len(rangeC))
#print(len(rangeD))
#rangeB.append(0)
#rangeC.append(0)
#rangeD.append(0)
print(min(rangeA)," + ",max(rangeA))
print(min(rangeB)," + ",max(rangeB))
print(min(rangeC)," + ",max(rangeC))
print(min(rangeD)," + ",max(rangeD))
#front = '/Users/42413/Documents/GitHub/EXAFS/Cu Data/path Data/feff'
front = '/Users/user/EXAFS-Physics/Cu Data/path Data/feff'
end = '.dat'

random.seed(19)
def generate_y():
    x_total = [0]*(401)
    y = []
    #print(y_total)
    for i in range(1,11):
        if i < 10:
            filename = front+'000'+str(i)+end
        elif i< 100:
            filename = front+'00'+str(i)+end
        else:
            filename = front+'0'+str(i)+end
        a = random.choice(rangeA)
        b = random.choice(rangeB)
        c = random.choice(rangeC)
        d = random.choice(rangeD)
        
        #print(filename)
        #print(a,b,c,d)
        path=feffdat.feffpath(filename, s02= str(a), e0= str(b), sigma2= str(d), deltar= str(c), _larch=mylarch)
        y.extend([a,b,d,c])
        #print(y)
        feffdat._path2chi(path, _larch=mylarch)
        x = path.chi
        #print("-------------------------------------------------------")
        #print(y)
        #print("------------------------------------------------------")
        for k in range(len(x)):
           x_total[k] += x[k]
    return(x_total, y)
x,y = generate_y()

#print(len(y))
#print(min(x), max(x))

x = []
y = []

for i in range(0,5000):
    X,Y = generate_y()
    x.append(X)
    y.append(Y)
#a = random.choice(rangeA)
#b = random.choice(rangeB)
#c = random.choice(rangeC)
#d = random.choice(rangeD) 
"""
## Check
k_value = (np.linspace(0,2000,401)*0.01).tolist()
k_value = np.asarray(k_value)
chi = y
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