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
from keras.models import load_model

def model_load(filename):
    model = load_model(filename)
    return model 

model = model_load('XAFS.h5')

mylarch = Interpreter()

g = read_ascii('/Users/shail/EXAFS-Physics/Cu Data/cu_10k.xmu', _larch = mylarch)
autobk(g,rbkg= 1.45,_larch = mylarch)

k_value = (np.linspace(0,2000,401)*0.01).tolist()
k_value = k_value[60:341]
chi = (g.chi).tolist()
chi = chi[60:341]
#chi.extend(chi_values) 
chi_y = np.asmatrix(chi)
k_value = np.asarray(k_value)
chi = np.asarray(chi)

chi_y = np. asmatrix(chi)

predict = model.predict(chi_y)

front = '/Users/shail/EXAFS-Physics/Cu Data/path Data/feff'
end = '.dat'

for i in range(1,11):
    #print(i)
    y_chi = []
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
    
y_chi = y_chi[0]
y_chi = y_chi[60:341]
#ch
import matplotlib.pyplot as plt
plt.plot(k_value,chi*k_value**2)
plt.plot(k_value,y_chi*k_value**2)

plt.show()

