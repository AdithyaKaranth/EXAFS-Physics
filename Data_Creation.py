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
S02 = (np.linspace(5,150,146) * 0.01).tolist()
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
def generate_x():
    x_total = [0]*(281)
    #print(y_total)
    path_sel = np.round(np.random.rand(10)).astype(int)
    for i in range(1,11):
        if i < 10:
            filename = front+'000'+str(i)+end
        elif i< 100:
            filename = front+'00'+str(i)+end
        else:
            filename = front+'0'+str(i)+end
        
        if path_sel[i-1] == 1:
            s02 = random.choice(S02)
            e0 = random.choice(E0)
            sigma2 = random.choice(SIGMA2)
            deltaR = random.choice(DELTAR)
            
        else:
            s02 = e0 = sigma2 = deltaR = 0
        
        #print(filename)
        #print(a,b,c,d)
        path=feffdat.feffpath(filename, s02= str(s02), e0= str(e0), sigma2= str(sigma2), deltar= str(deltaR), _larch=mylarch)
        #print(y)
        feffdat._path2chi(path, _larch=mylarch)
        x = path.chi
        x = x[60:341]
        #print("-------------------------------------------------------")
        #print(y)
        #print("------------------------------------------------------")
        for k in range(len(x)):
           x_total[k] += x[k]
    return x_total

#print(len(y))
#print(min(x), max(x))

x = []

for i in range(0,5000):
    X = generate_x()
    x.append(X)

#a = random.choice(rangeA)
#b = random.choice(rangeB)
#c = random.choice(rangeC)
#d = random.choice(rangeD) 
    
k_value = (np.linspace(0,2000,401)*0.01).tolist()
k_value = np.asarray(k_value)
k_value = k_value[60:341]
chi = np.asarray(x[4996])
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(k_value,chi*k_value**2)

plt.show()

"""
path=feffdat.feffpath('/Users/user/EXAFS-master/Cu Data/path Data/feff0001.dat', s02= str(rangeA[20]), e0= str(rangeB[20]), sigma2= rangeD[20], deltar= rangeC[20], _larch=mylarch)
feffdat._path2chi(path, _larch=mylarch)
y = path.chi
#print(y)
"""
import numpy as np
x = np.asarray(x)

import pickle

with open('input.pickle', 'wb') as f:
    pickle.dump(x, f,)