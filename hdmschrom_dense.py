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
from keras import regularizers

import pickle

def HDMSDenseNet(input_shape, num_classes=2):
    
    net = {}
    # input
    input_tensor = Input(shape=input_shape, name='input_11')
    net['input_11'] = input_tensor
    net['reshape_11'] = Reshape((input_shape[0], input_shape[1], 1), name='reshape_11')(net['input_11'])
    
    # encoding
    net['conv2d_11'] = Conv2D(32, (3, 3), padding='same', name='conv2d_11')(net['reshape_11'])
    net['batchnorm_11'] = BatchNormalization(name='batchnorm_11')(net['conv2d_11'])
    net['relu_11'] = Activation(activation='relu', name='relu_11')(net['batchnorm_11'])
    net['conv2d_12'] = Conv2D(32, (3, 3), padding='same', name='conv2d_12')(net['relu_11'])
    net['batchnorm_12'] = BatchNormalization(name='batchnorm_12')(net['conv2d_12'])
    net['relu_12'] = Activation(activation='relu', name='relu_12')(net['batchnorm_12'])

    net['maxpooling2d_1'] = MaxPooling2D(name='maxpooling2d_1')(net['relu_12'])

    net['conv2d_21'] = Conv2D(64, (3, 3), padding='same', name='conv2d_21')(net['maxpooling2d_1'])
    net['batchnorm_21'] = BatchNormalization(name='batchnorm_21')(net['conv2d_21'])
    net['relu_21'] = Activation(activation='relu', name='relu_21')(net['batchnorm_21'])
    net['conv2d_22'] = Conv2D(64, (3, 3), padding='same', name='conv2d_22')(net['relu_21'])
    net['batchnorm_22'] = BatchNormalization(name='batchnorm_22')(net['conv2d_22'])
    net['relu_22'] = Activation(activation='relu', name='relu_22')(net['batchnorm_22'])
    net['conv2d_23'] = Conv2D(64, (3, 3), padding='same', name='conv2d_23')(net['relu_22'])
    net['batchnorm_23'] = BatchNormalization(name='batchnorm_23')(net['conv2d_23'])
    net['relu_23'] = Activation(activation='relu', name='relu_23')(net['batchnorm_23'])

    net['maxpooling2d_2'] = MaxPooling2D(name='maxpooling2d_2')(net['relu_23'])

    net['conv2d_31'] = Conv2D(128, (3, 3), padding='same', name='conv2d_31')(net['maxpooling2d_2'])
    net['batchnorm_31'] = BatchNormalization(name='batchnorm_31')(net['conv2d_31'])
    net['relu_31'] = Activation(activation='relu', name='relu_31')(net['batchnorm_31'])
    net['conv2d_32'] = Conv2D(128, (3, 3), padding='same', name='conv2d_32')(net['relu_31'])
    net['batchnorm_32'] = BatchNormalization(name='batchnorm_32')(net['conv2d_32'])
    net['relu_32'] = Activation(activation='relu', name='relu_32')(net['batchnorm_32'])
    net['conv2d_33'] = Conv2D(1, (3, 3), padding='same', name='conv2d_33')(net['relu_32'])
    net['batchnorm_33'] = BatchNormalization(name='batchnorm_33')(net['conv2d_33'])
    net['relu_33'] = Activation(activation='relu', name='relu_33')(net['batchnorm_33'])

    # decoding...
    net['upsampling2d_1'] = UpSampling2D(name='upsampling2d_1')(net['relu_33'])

    net['conv2d_71'] = Conv2D(64, (3, 3), padding='same', name='conv2d_71')(net['upsampling2d_1'])
    net['batchnorm_71'] = BatchNormalization(name='batchnorm_71')(net['conv2d_71'])
    net['relu_71'] = Activation(activation='relu', name='relu_71')(net['batchnorm_71'])
    net['conv2d_72'] = Conv2D(64, (3, 3), padding='same', name='conv2d_72')(net['relu_71'])
    net['batchnorm_72'] = BatchNormalization(name='batchnorm_72')(net['conv2d_72'])
    net['relu_72'] = Activation(activation='relu', name='relu_72')(net['batchnorm_72'])
    net['conv2d_73'] = Conv2D(64, (3, 3), padding='same', name='conv2d_73')(net['relu_72'])
    net['batchnorm_73'] = BatchNormalization(name='batchnorm_73')(net['conv2d_73'])
    net['relu_73'] = Activation(activation='relu', name='relu_73')(net['batchnorm_73'])

    net['upsampling2d_2'] = UpSampling2D(name='upsampling2d_2')(net['relu_73'])

    net['conv2d_81'] = Conv2D(32, (3, 3), padding='same', name='conv2d_81')(net['upsampling2d_2'])
    net['batchnorm_81'] = BatchNormalization(name='batchnorm_81')(net['conv2d_81'])
    net['relu_81'] = Activation(activation='relu', name='relu_81')(net['batchnorm_81'])
    net['conv2d_82'] = Conv2D(32, (3, 3), padding='same', name='conv2d_82')(net['relu_81'])
    net['batchnorm_82'] = BatchNormalization(name='batchnorm_82')(net['conv2d_82'])
    net['relu_82'] = Activation(activation='relu', name='relu_82')(net['batchnorm_82'])
    net['conv2d_83'] = Conv2D(32, (3, 3), padding='same', name='conv2d_83')(net['relu_82'])
    net['batchnorm_83'] = BatchNormalization(name='batchnorm_83')(net['conv2d_83'])
    net['relu_83'] = Activation(activation='sigmoid', name='relu_83')(net['batchnorm_83'])

    # Decoded
    net['decoded'] = net['relu_83']

    # Class Prediction

    net['flatten_1'] = Flatten(name='flatten_1')(net['relu_33'])

    net['dense_101'] = Dense(64, name='dense_101', kernel_regularizer=regularizers.l2(0.001))(net['flatten_1'])
    net['batchnorm_101'] = BatchNormalization(name='batchnorm_101')(net['dense_101'])
    net['activation_101'] = Activation(activation='relu', name='activation_101')(net['batchnorm_101'])
    net['dense_102'] = Dense(64, name='dense_102', kernel_regularizer=regularizers.l2(0.001))(net['activation_101'])
    net['batchnorm_102'] = BatchNormalization(name='batchnorm_102')(net['dense_102'])
    net['activation_102'] = Activation(activation='relu', name='activation_102')(net['batchnorm_102'])
    net['dense_103'] = Dense(64, name='dense_103', kernel_regularizer=regularizers.l2(0.001))(net['activation_102'])
    net['batchnorm_103'] = BatchNormalization(name='batchnorm_103')(net['dense_103'])
    net['activation_103'] = Activation(activation='relu', name='activation_103')(net['batchnorm_103'])

    net['class_prediction'] = Dense(2, name='class_prediction', activation='softmax')(net['activation_103'])

    model = Model(net['input_11'], [net['class_prediction'], net['decoded']])
    return model

if __name__ == '__main__':
    input_shape = (192, 1024, )

    mymodel = HDMSDenseNet(input_shape)
    print(mymodel.summary())
