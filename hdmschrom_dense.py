"""Keras implementation of SSD."""
import numpy as np
import keras.backend as K
if K.backend() == 'cntk':
    import cntk as C
import keras
from keras.layers import Activation
from keras.layers import AtrousConvolution2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import BatchNormalization
from keras.layers import merge
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model

from mschromnet_layers import PriorBox
from mschromnet_layers import Normalize
from mschromnet_layers import MagnifyAndClip
from mschromnet_layers import LogTransform

from mschromnet_utils import BBoxUtility
from gdrivegenerator import GdriveGenerator
import pickle

def HDMSDenseNet(input_shape, num_classes=2):
    
    net = {}
    # input
    input_tensor = Input(shape=input_shape, name='input_1')
    net['input_1'] = input_tensor
    net['flatten_1'] = Flatten(name='flatten_1')(net['input_1'])  

    # Class Prediction
    net['dense_1'] = Dense(64, name='dense_1', activation='relu')(net['flatten_1'])
    net['dense_2'] = Dense(64, name='dense_2', activation='relu')(net['dense_1'])
    net['dense_3'] = Dense(64, name='dense_3', activation='relu')(net['dense_2'])
    net['class_prediction'] = Dense(2, name ='class_prediction', activation='softmax')(net['dense_3'])

    model = Model(net['input_1'], net['class_prediction'])
    return model

if __name__ == '__main__':
    input_shape = (192, 1024, )

    mymodel = HDMSDenseNet(input_shape)
    print(mymodel.summary())
