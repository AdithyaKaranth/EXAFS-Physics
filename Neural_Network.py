# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:49:08 2019

@author: Adithya Karantha
"""
from keras import models
from keras import layers
from keras import metrics
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.layers import Activation
from keras.models import Model
import larch
from larch_plugins.xafs import feffdat

from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def path_activation(x):
    return (K.tanh(x) *5)

get_custom_objects().update({'path_activation': Activation(path_activation)})

import pickle

with open('input.pickle', 'rb') as f:
    x = pickle.load(f)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(200,input_shape = (281,)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(layers.Dense(40,name ="path_layer"))
    model.add(Activation(path_activation))
    
    model.add(layers.Dense(200))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    
    model.add(layers.Dense(281))
    
    model.compile(optimizer=optimizers.Adam(lr = 0.001),loss= losses.mean_squared_error ,metrics =[metrics.mse])
    return model

model = build_model()



history = model.fit(x,x,epochs = 80,batch_size= 16,validation_split=0.2)

history_dict = history.history
#print(history_dict)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)

import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(epochs,loss_values,'bo',label = 'Training Loss')
plt.plot(epochs,val_loss_values,'b',label = 'Validation Loss')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#def save_model(model):
    #model.save('XAFS.h5')

#save_model(model)
from larch import Interpreter
from larch_plugins.io import read_ascii
from larch_plugins.xafs import autobk
import numpy as np

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
