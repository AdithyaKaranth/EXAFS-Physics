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

from keras.layers.normalization import BatchNormalization

from keras import regularizers


import pickle

with open('input.pickle', 'rb') as f:
    x = pickle.load(f)

with open('output.pickle', 'rb') as f:
    y = pickle.load(f)


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(300,activation = "relu",input_shape = (401,)))
    
    model.add(BatchNormalization())
    
    model.add(layers.Dense(40,activation = "relu"))
    model.compile(optimizer=optimizers.Adam(lr = 0.0001),loss= losses.mean_squared_error ,metrics =[metrics.mse])
    return model

model = build_model()



history = model.fit(x,y,epochs = 80,batch_size=16,validation_split=0.2)

history_dict = history.history
#print(history_dict)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)

import matplotlib.pyplot as plt

plt.plot(epochs,loss_values,'bo',label = 'Training Loss')
plt.plot(epochs,val_loss_values,'b',label = 'Validation Loss')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

def save_model(model):
    model.save('XAFS.h5')

save_model(model)