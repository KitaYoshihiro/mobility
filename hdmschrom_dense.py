"""Keras implementation of SSD."""
import numpy as np
import keras.backend as K
if K.backend() == 'cntk':
    import cntk as C
import keras
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import BatchNormalization
from keras.layers import merge
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model

import pickle

def HDMSDenseNet(input_shape, num_classes=2):
    
    net = {}
    # input
    input_tensor = Input(shape=input_shape, name='input_1')
    net['input_1'] = input_tensor
    net['flatten_1'] = Flatten(name='flatten_1')(net['input_1'])  

    # Class Prediction
    net['dense_1'] = Dense(64, name='dense_1')(net['flatten_1'])
    net['batchnorm_1'] = BatchNormalization(name='batchnorm_1')(net['dense_1'])
    net['activation_1'] = Activation(activation='relu', name='activation_1')(net['batchnorm_1'])
    net['dense_2'] = Dense(64, name='dense_2')(net['activation_1'])
    net['batchnorm_2'] = BatchNormalization(name='batchnorm_2')(net['dense_2'])
    net['activation_2'] = Activation(activation='relu', name='activation_2')(net['batchnorm_2'])
    net['dense_3'] = Dense(64, name='dense_3')(net['activation_2'])
    net['batchnorm_3'] = BatchNormalization(name='batchnorm_3')(net['dense_3'])
    net['activation_3'] = Activation(activation='relu', name='activation_3')(net['batchnorm_3'])

    net['class_prediction'] = Dense(2, name ='class_prediction', activation='softmax')(net['activation_3'])

    model = Model(net['input_1'], net['class_prediction'])
    return model

if __name__ == '__main__':
    input_shape = (192, 1024, )

    mymodel = HDMSDenseNet(input_shape)
    print(mymodel.summary())
