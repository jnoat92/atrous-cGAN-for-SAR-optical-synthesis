#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:45:09 2019

@author: daliana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:29:39 2019

@author: daliana
"""

# todo upgrade to keras 2.0
from keras.models import Model
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

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

    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    conv_3 = Convolution2D(64, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(64, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(128, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(128, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    
#     Camada agregada
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_7)

    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(128, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(128, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
#

    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_7)
    
    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(pool_3)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(128, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(128, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_7)
#
    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_3)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_7)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5 = MaxPooling2D(pool_size=(2, 2))(conv_13)
    
    # decoder

    unpool_1 = UpSampling2D(pool_size)(pool_5)

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(256, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = UpSampling2D(pool_size)(conv_16)

    conv_17 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(128, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(128, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)
    
    unpool_3= UpSampling2D(pool_size)(conv_19)

    conv_17 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(128, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(128, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)
    
#    Agreagada
    unpool_4 = UpSampling2D(pool_size)(conv_19)
    
    conv_17 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_4)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_17 = Convolution2D(128, (kernel, kernel), padding="same")(conv_17)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(128, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    
    unpool_5 = UpSampling2D(pool_size)(conv_18)
    
    conv_20 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(64, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    
    unpool_6 = UpSampling2D(pool_size)(conv_21)
    
    conv_22 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_6)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)
    conv_23 = Convolution2D(64, (kernel, kernel), padding="same")(conv_22)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)

    unpool_5 = UpSampling2D(pool_size)(conv_23)

    conv_22 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_6)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)
    conv_23 = Convolution2D(64, (kernel, kernel), padding="same")(conv_22)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)

    unpool_5 = UpSampling2D(pool_size)(conv_23)
    
    conv_25 = Convolution2D(32, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(2, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)

    outputs = Activation('softmax')(conv_26)

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return model
   
if __name__ == "__main__":
    model = Segnet(nClasses = 2, optimizer = None, input_width = 256 , input_height = 256 , nChannels = 3)
    model.summary()