from __future__ import print_function, division
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import SGD,Adam

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from utils import callParzen
from cgan import CGAN
from cgan import loadmodel
cgan = CGAN()
cgan.train(50000,100,200)

# cgan=loadmodel(epoch=500,path=".././models") #Uncomment to load model from models
print(callParzen(cgan,100))