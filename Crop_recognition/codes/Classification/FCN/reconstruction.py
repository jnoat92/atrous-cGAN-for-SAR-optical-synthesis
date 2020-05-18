#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:40:28 2020

@author: daliana
"""
import numpy as np
import cv2

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def gray2rgb(image):
    """
    Funtion to convert classes values from 0,1,3,4 to rgb values
    """
    row,col = image.shape
    image = image.reshape((row*col))
    rgb_output = np.zeros((row*col, 3))
    rgb_map = [[0,0,255],[0,255,0],[0,255,255],[255,255,0],[255,255,255]]
    for j in np.unique(image):
        rgb_output[image==j] = np.array(rgb_map[j])
    
    rgb_output = rgb_output.reshape((row,col,3))  
    rgb_output = cv2.cvtColor(rgb_output.astype('uint8'),cv2.COLOR_BGR2RGB)
    return rgb_output 

class Image_reconstruction(object):

    def __init__ (self, net, output_c_dim, patch_size=512, overlap_percent=0):

        self.net = net
        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.output_c_dim = output_c_dim
   
    def Inference(self, tile):
       
        '''
        Normalize before call this method
        '''

        num_rows, num_cols, _ = tile.shape

        # Percent of overlap between consecutive patches.
        # The overlap will be multiple of 2
        overlap = round(self.patch_size * self.overlap_percent)
        overlap -= overlap % 2
        stride = self.patch_size - overlap
       
        # Add Padding to the image to match with the patch size and the overlap
        step_row = (stride - num_rows % stride) % stride
        step_col = (stride - num_cols % stride) % stride
 
        pad_tuple = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)), (0,0) )
        tile_pad = np.pad(tile, pad_tuple, mode = 'symmetric')

        # Number of patches: k1xk2
        k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
        print('Number of patches: %d x %d' %(k1, k2))

        # Inference
        probs = np.zeros((k1*stride, k2*stride, self.output_c_dim))

        for i in range(k1):
            for j in range(k2):
               
                patch = tile_pad[i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size), :]
                patch = patch[np.newaxis,...]
                infer = self.net.predict(patch, verbose=0)

                probs[i*stride : i*stride+stride,
                      j*stride : j*stride+stride, :] = infer[0, overlap//2 : overlap//2 + stride,
                                                                overlap//2 : overlap//2 + stride, :]
            print('row %d' %(i+1))

        # Taken off the padding
        probs = probs[:k1*stride-step_row, :k2*stride-step_col]

        return probs
    

    
    