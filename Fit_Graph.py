# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:33:11 2019

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
from larch_plugins.std import show
from keras.models import load_model
from keras.models import Model

def model_load(filename):
    model = load_model(filename)
    return model 

model = model_load('XAFS.h5')

mylarch = Interpreter()

g = read_ascii('/Users/user/EXAFS-master/Cu Data/cu_10k.xmu', _larch = mylarch)
autobk(g, rbkg= 1.45, _larch = mylarch)
#show(g,_larch = mylarch)

k_value = (np.linspace(0,2000,401)*0.01).tolist()
k_value = k_value[60:341]
chi = (g.chi).tolist()
chi = chi[60:341] 

k_value = np.asarray(k_value)
chi = np.asarray(chi)

chi_y = np. asmatrix(chi)

intermediate_layer_model = Model(input=model.input, output=model.get_layer("path_layer").output)
predict = intermediate_layer_model.predict(chi_y)
#predict = model.predict(chi_y)

front = '/Users/user/EXAFS-Physics/Cu Data/path Data/feff'
end = '.dat'

for i in range(1,11):
    print(i)
    y_chi = [0]*281
    if i < 10:
        filename = front+'000'+str(i)+end
    elif i< 100:
        filename = front+'00'+str(i)+end
    else:
        filename = front+'0'+str(i)+end
            
    path=feffdat.feffpath(filename, s02=str(predict[0][4*(i-1)]) , e0= str(predict[0][4*(i-1) + 1]), sigma2= str(predict[0][4*(i-1) + 2]), deltar= str(predict[0][4*(i-1) +3]), _larch=mylarch)
    feffdat._path2chi(path, _larch=mylarch)
    
    y = path.chi
    y = y[60:341]
    y_chi += y

print(len(y_chi))
#g = read_ascii('/Users/user/EXAFS-master/Cu Data/cu_10k.xmu', _larch = mylarch)
#ch
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
plt.clf()
#plt.plot(k_value,chi*k_value**2)
plt.plot(k_value,y_chi*k_value**2)

plt.show()

