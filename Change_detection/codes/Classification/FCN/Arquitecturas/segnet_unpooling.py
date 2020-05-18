# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:18:16 2019

@author: pmad9589
"""

#%%
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
import matplotlib.image as mpimg
from Arquitecturas.layer_segnet import MaxPoolingWithArgmax2D, MaxUnpooling2D

def Segnet (nClasses = 2, optimizer = None, input_width = 512 , input_height = 512 , nChannels = 3): 

    input_shape = [input_height, input_width, nChannels]
    inputs = Input(input_shape)
    kernel=3
    pool_size=(2, 2)

    conv_1 = Convolution2D(32, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(32, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(64, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(64, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(128, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(128, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

#    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)
#
#    conv_8 = Convolution2D(256, (kernel, kernel), padding="same")(pool_3)
#    conv_8 = BatchNormalization()(conv_8)
#    conv_8 = Activation("relu")(conv_8)
#    conv_9 = Convolution2D(256, (kernel, kernel), padding="same")(conv_8)
#    conv_9 = BatchNormalization()(conv_9)
#    conv_9 = Activation("relu")(conv_9)
#    conv_10 = Convolution2D(256, (kernel, kernel), padding="same")(conv_9)
#    conv_10 = BatchNormalization()(conv_10)
#    conv_10 = Activation("relu")(conv_10)
#
#    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)
#
#    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
#    conv_11 = BatchNormalization()(conv_11)
#    conv_11 = Activation("relu")(conv_11)
#    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
#    conv_12 = BatchNormalization()(conv_12)
#    conv_12 = Activation("relu")(conv_12)
#    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
#    conv_13 = BatchNormalization()(conv_13)
#    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_7)
    print("Build encoder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])
#
#    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
#    conv_14 = BatchNormalization()(conv_14)
#    conv_14 = Activation("relu")(conv_14)
#    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
#    conv_15 = BatchNormalization()(conv_15)
#    conv_15 = Activation("relu")(conv_15)
#    conv_16 = Convolution2D(256, (kernel, kernel), padding="same")(conv_15)
#    conv_16 = BatchNormalization()(conv_16)
#    conv_16 = Activation("relu")(conv_16)
#
#    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])
#
#    conv_17 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_2)
#    conv_17 = BatchNormalization()(conv_17)
#    conv_17 = Activation("relu")(conv_17)
#    conv_18 = Convolution2D(256, (kernel, kernel), padding="same")(conv_17)
#    conv_18 = BatchNormalization()(conv_18)
#    conv_18 = Activation("relu")(conv_18)
#    conv_19 = Convolution2D(128, (kernel, kernel), padding="same")(conv_18)
#    conv_19 = BatchNormalization()(conv_19)
#    conv_19 = Activation("relu")(conv_19)
#
#    unpool_3 = MaxUnpooling2D(pool_size)([conv_7, mask_5])

    conv_20 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_1)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(128, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(64, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(32, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(32, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(nClasses, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)


    outputs = Activation('softmax')(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return model

#%%
if __name__ == "__main__":
    model = Segnet(nClasses = 3,optimizer = None, input_width = 128 , input_height = 128 , nChannels = 16)
    model.summary()
#    for i in model.layers:
#        print(i.name)
#    data = mpimg.imread('DJI_0001.JPG')
#    data = data.reshape(1,data.shape[0],data.shape[1],data.shape[2])

#    outputs = [model.layers[45].output]  
#    functor = K.function([model.input, K.learning_phase()], outputs)
#    layer_outs = functor([data, 1.])
#    print(layer_outs)
    
