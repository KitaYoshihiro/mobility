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

def Conv2DBNRelu(input, net, basename, cn):
    """CNN2D
    """
    net['conv' + basename] = Conv2D(cn, (3, 3), padding='same', name='conv' + basename)(input)
    net['norm' + basename] = BatchNormalization(name='norm' + basename)(net['conv' + basename])
    net['relu' + basename] = Activation(activation='relu', name='relu' + basename)(net['norm' + basename])
    return net['relu' + basename]

def Conv2DBNSigmoid(input, net, basename, cn):
    """CNN2D
    """
    net['conv' + basename] = Conv2D(cn, (3, 3), padding='same', name='conv' + basename)(input)
    net['norm' + basename] = BatchNormalization(name='norm' + basename)(net['conv' + basename])
    net['sigmoid' + basename] = Activation(activation='sigmoid', name='sigmoid' + basename)(net['norm' + basename])
    return net['sigmoid' + basename]

def MaxPool2D(input, net, basename):
    """MaxPool2D
    """
    net['pool' + basename] = MaxPooling2D(name='pool' + basename)(input)
    return net['pool' + basename]

def Upsample2D(input, net, basename):
    """Upsample2D
    """
    net['upsample' + basename] =  UpSampling2D(name='upsample' + basename)(input)
    return net['upsample' + basename]

def Concat(input, input2, net, basename):
    """Concat (Cocatenation axis = 3 for 2D data)
    """
    net['concat' + basename] =  Concatenate(name='concat' + basename, axis=3)([input, input2])
    return net['concat' + basename] 

def UNet_Builder(input, net, initial_layer_id, structure, depth=20, u_net=True):
    """ building U-net
    # input: input keras tensor
    # net: list for network layers (keras tensors)
    # initial_layer_id: int value for starting layer number (for example, initial_layer_id=2)
    # structure: 2D-array of channels (example: structure = [[16,16],[32,32,32],[64,64,64]])
    """
    structure_length = len(structure)
    if depth > structure_length or depth <= 1:
        depth = structure_length
    
    if depth != structure_length:
        depth_label = '_d' + str(depth)
    else:
        depth_label = ''
    
    if u_net:
        depth_label += 'u'

    while len(structure) > depth:
        structure.pop()

    channels = structure.pop(0)
    x = input
    subid = 1
    for channel in channels:
        x = Conv2DBNRelu(x, net, '_f_'+str(initial_layer_id)+'_'+str(subid), channel)
        subid += 1
    # if len(structure) == 0:
    #    return x # 最下層はここでリターン！
    # xx = x
    if len(structure) > 0:
        xx = x
        x = MaxPool2D(x, net, '_f_'+str(initial_layer_id))
        initial_layer_id +=1
        x = UNet_Builder(x, net, initial_layer_id, structure, u_net=u_net)
        initial_layer_id -=1
        x = Upsample2D(x, net, '_r_'+str(initial_layer_id)+depth_label)
        if u_net:
            x = Concat(xx, x, net, '_r_'+str(initial_layer_id)+depth_label) 
        subid = 1
        for channel in reversed(channels):
            x = Conv2DBNRelu(x, net, '_r_'+str(initial_layer_id)+'_'+str(subid)+depth_label, channel)
            subid += 1
    return x

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
