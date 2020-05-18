#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:31:37 2019

@author: daliana
"""
from keras.models import Model
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, core, Conv2DTranspose

def Unet (nClasses = 3, input_width = 128 , input_height = 128 , nChannels = 16): 
    
    inputs = Input((input_height, input_width, nChannels))
    # contracting path
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv4)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#    
#    conv5 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool4)
#    conv5 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv5)
#    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
#    
#    conv6 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool5)
#    conv6 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv6)
#    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
#    
#    conv7 = Convolution2D(256, (3, 3), activation='relu', padding='same')(pool6)
#    conv7 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv7)
#    pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)
#    
#    conv8 = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool7)
#    conv8 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv8)
    
    # expansive path
#    up1 =  Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same') (conv8)
#    merge1 = concatenate([conv7,up1], axis = 3)
#    conv9 = Convolution2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
#    conv9 = Convolution2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#    
#    up2 =  Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same') (conv9)
#    merge2 = concatenate([conv6,up2], axis = 3)
#    conv10 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
#    conv10 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
#    
#    up3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same') (conv10)
#    merge3 = concatenate([conv5,up3], axis = 3)
#    conv11 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
#    conv11 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
#
#    up4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same') (conv11)
#    merge4 = concatenate([conv4,up4], axis = 3)
#    conv12 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
#    conv12 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    
    up5 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same') (conv4)
    merge5 = concatenate([conv3,up5], axis = 3)
    conv13 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv13 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv13)
    
    up6 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (conv13)
    merge6 = concatenate([conv2,up6], axis = 3)
    conv14 = Convolution2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv14 = Convolution2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv14)

    up7 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same') (conv14)
    merge7 = concatenate([conv1,up7], axis = 3)
    conv15 = Convolution2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv15 = Convolution2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv15)
    
    conv16 = Convolution2D(nClasses, (1, 1), activation='relu',padding='same')(conv15)
    
    conv17 = core.Activation('softmax')(conv16)

    model = Model(inputs, conv17)

    return model

if __name__ == "__main__":
    model = Unet()
    model.summary()