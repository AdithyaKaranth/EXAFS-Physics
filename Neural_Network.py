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
from keras.layers.normalization import BatchNormalization
import larch
from keras import regularizers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import pickle

with open('input.pickle', 'rb') as f:
    x = pickle.load(f)

with open('output.pickle', 'rb') as f:
    y = pickle.load(f)
 

def path_activation(x):
    return (K.tanh(x) * 5)

get_custom_objects().update({'path_activation': Activation(path_activation)})


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(200,activation = "tanh",input_shape = (281,)))
    
    #model.add(BatchNormalization())
    
    model.add(layers.Dense(40,activation = Activation(path_activation)))
    model.compile(optimizer=optimizers.Adam(lr = 0.0001),loss= losses.mean_squared_error ,metrics =[metrics.mse])
    return model

model = build_model()



history = model.fit(x,y,epochs = 100,batch_size=16,validation_split=0.2)

history_dict = history.history
#print(history_dict)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

plt.plot(epochs,loss_values,'bo',label = 'Training Loss')
plt.plot(epochs,val_loss_values,'b',label = 'Validation Loss')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""def save_model(model):
    model.save('XAFS.h5')

save_model(model)"""

from larch import Interpreter
mylarch = Interpreter()


import numpy as np

k_value = (np.linspace(0,2000,401)*0.01).tolist()
k_value = k_value[60:341]
chi = x[89].tolist()

#chi.extend(chi_values) 
chi_y = np.asmatrix(chi)
k_value = np.asarray(k_value)

actual_chi = np.asarray(chi)

chi_y = np.asmatrix(chi)

#plt.plot(k_value,chi*k_value**2)

from larch_plugins.xafs import feffdat
from larch import Interpreter

predict = model.predict(chi_y)

front = '/Users/user/EXAFS-Physics/Cu Data/path Data/feff'
end = '.dat'

for i in range(1,11):
    #print(i)
    predicted_chi = [0]*281
    if i < 10:
        filename = front+'000'+str(i)+end
    elif i< 100:
        filename = front+'00'+str(i)+end
    else:
        filename = front+'0'+str(i)+end
            
    path=feffdat.feffpath(filename, s02=str(predict[0][4*(i-1)]) , e0= str(predict[0][4*(i-1) + 1]), sigma2= str(predict[0][4*(i-1) + 2]), deltar= str(predict[0][4*(i-1) +3]), _larch=mylarch)
    feffdat._path2chi(path, _larch=mylarch)
    
    chi = path.chi
    chi = chi[60:341]
    predicted_chi += chi

plt.plot(k_value,actual_chi*k_value**2)
plt.plot(k_value,predicted_chi*k_value**2)
plt.show()