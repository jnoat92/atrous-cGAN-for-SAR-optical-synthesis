#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:48:05 2019

@author: daliana
"""


from keras.layers import Activation,MaxPooling2D,UpSampling2D,Dense,BatchNormalization,Input,Reshape,multiply,add,Dropout,AveragePooling2D,GlobalAveragePooling2D,concatenate, Lambda,Concatenate
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.models import Model	
from keras import layers													  
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer,InputSpec
from keras.utils import conv_utils
import numpy as np
import sys
from layers_fusion import TransitionDown, TransitionUp, SoftmaxLayer,SepConv_BN
#from deeplabv3p import _xception_block, SepConv_BN, _conv2d_same

def Tiramisu(
        input_shape=(512,512,3),
        n_classes = 1,
        n_filters_first_conv = 48,
        n_pool = 5,
        growth_rate = 16 ,
        n_layers_per_block = [4,5,7,10,12,15,12,10,7,5,4],
        dropout_p = 0.2
        ):
    if type(n_layers_per_block) == list:
            print(len(n_layers_per_block))
    elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError
        
#####################
# First Convolution #
#####################        
    inputs = Input(shape=input_shape)
    residual_output = Conv2D(filters=n_filters_first_conv, kernel_size=3, padding='same', kernel_initializer='he_uniform')(inputs)
    filters = n_filters_first_conv

#####################
# Downsampling path #
#####################     
    skip_connection_list = []
    idb = 0
    for i in range(n_pool):
        stack = residual_output
        for j in range(n_layers_per_block[i]):
            
            l = SepConv_BN(residual_output, dropout_p, filters = growth_rate, prefix = 'layers_separable_conv{}'.format(idb),
                           stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
            shortcut = Conv2D(filters = growth_rate, kernel_size = 1)(residual_output)
            residual_output = add([shortcut, l])
            filters +=  growth_rate
            
            idb +=1  
            stack = concatenate([stack, l])
        # At the end of the dense block, the current stack is stored in the skip_connections list
#        print(stack.shape)
        skip_connection_list.append(stack)
        residual_output = TransitionDown(stack, filters, dropout_p, idb)
        
    skip_connection_list = skip_connection_list[::-1]
#    sys.exit()

#####################
#    Bottleneck     #
#####################  
    
    block_to_upsample=[]

#   This is for the Conventional DenseNet with separable convolution
    
    stack = residual_output
    for j in range(n_layers_per_block[n_pool]):
        
        l = SepConv_BN(residual_output, dropout_p, filters = growth_rate, prefix = 'layers_separable_conv{}'.format(idb),
                       stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
        shortcut = Conv2D(filters = growth_rate, kernel_size = 1)(residual_output)
        residual_output = add([shortcut, l])
        block_to_upsample.append(l)
        idb +=1  
        stack = concatenate([stack, l])
        
    block_to_upsample = concatenate(block_to_upsample)

   
#    This is if I using the atrous separable pyramid pooling
#    OS = 16
#    atrous_rates = (6, 12, 18)
#
#    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(stack)
#        
#    b4 = Conv2D(256, (1, 1), padding='same',
#                use_bias=False, name='image_pooling')(b4)
#    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
#    b4 = Activation('relu')(b4)
#    
#    b4 = Lambda(lambda x: K.tf.image.resize_bilinear(x,size=(int(np.ceil(input_shape[0]/OS)), int(np.ceil(input_shape[1]/OS)))))(b4)
#
#    # simple 1x1
#    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(stack)
#    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
#    b0 = Activation('relu', name='aspp0_activation')(b0)
#
#    # there are only 2 branches in mobilenetV2. not sure why
#        # rate = 6 (12)
#    b1 = SepConv_BN(stack, dropout_p, 256, 'aspp1',
#                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
#    
#    # rate = 12 (24)
#    b2 = SepConv_BN(stack, dropout_p, 256, 'aspp2',
#                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
#    # rate = 18 (36)
#    b3 = SepConv_BN(stack, dropout_p, 256, 'aspp3',
#                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
#
#    # concatenate ASPP branches & project
#    block_to_upsample = Concatenate()([b4, b0, b1, b2, b3])
#    
#    block_to_upsample = Conv2D(256, (1, 1), padding='same',
#               use_bias=False, name='concat_projection')(block_to_upsample)
#    block_to_upsample = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(block_to_upsample)
#    block_to_upsample = Activation('relu')(block_to_upsample)
#    print(block_to_upsample.shape)

   
#####################
#  Upsampling path  #
#####################

    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i ]
        print(n_filters_keep)
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)
        
        residual_output = stack
        block_to_upsample = []
        
        for j in range(n_layers_per_block[ n_pool + i + 1 ]):
            
            l = SepConv_BN(residual_output, dropout_p, filters = growth_rate, prefix = 'layers_separable_conv{}'.format(idb),
                           stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
            shortcut = Conv2D(filters = growth_rate, kernel_size = 1)(residual_output)
            residual_output = add([shortcut, l])
            block_to_upsample.append(l)
            idb +=1  
            stack = concatenate([stack, l])
        block_to_upsample = concatenate(block_to_upsample)


#####################
#  Softmax          #
#####################
    output = SoftmaxLayer(stack, n_classes)            
    model=Model(inputs = inputs, outputs = output)    
    
    return model
    
if __name__ == "__main__":
    model = Tiramisu(input_shape=(512,512,3), n_classes = 2, n_filters_first_conv = 48, growth_rate = 16 , n_pool = 5, n_layers_per_block = [4,5,7,10,12,15,12,10,7,5,4], dropout_p = 0.2)
#    model = Tiramisu(input_shape=(512,512,3), n_classes = 2, n_filters_first_conv = 32, n_pool = 8 , growth_rate = 8 , n_layers_per_block = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], dropout_p = 0)
    model.summary()
