"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import tensorflow as tf
from time import gmtime, strftime
# from osgeo import gdal
# import tifffile as tiff
from PIL import Image
import glob
from skimage.transform import resize
from skimage import exposure
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
import keras
import collections  
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.externals import joblib
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import sys
from skimage.util.shape import view_as_windows
import itertools
import multiprocessing
from time import sleep
from functools import partial

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_tiff_image(patch):
    print( patch)
    img = np.array(Image.open(patch))
    return img

def Normalization(img, mask, norm_type, scaler_name="scaler"):

    num_rows, num_cols, bands = img.shape
    img = img.reshape(num_rows * num_cols, bands)

    if norm_type == 'min_max':
        scaler = pre.MinMaxScaler((-1, 1)).fit(img[mask.ravel() == 1])
        print('min_max normalization!!!')
    elif norm_type == 'std':
        scaler = pre.StandardScaler().fit(img[mask.ravel() == 1])
        print('std normalization!!!')
    elif norm_type == 'wise_frame_mean':
        scaler = pre.StandardScaler(with_std=False).fit(img[mask.ravel() == 1])
        print('wise_frame_mean normalization!!!')
    else:
        print('without normalization!!!')
        img = img.reshape(num_rows, num_cols, bands)
        return img

    # save scaler
    joblib.dump(scaler, scaler_name  + '_' + norm_type + '.pkl')
    img = np.float32(scaler.transform(img))
    img = img.reshape(num_rows, num_cols, bands)

    return img

def Denormalization(img, scaler):
    rows, cols, _ = img.shape
    img = img.reshape((rows * cols, -1))
    img = scaler.inverse_transform(img)
    img = img.reshape((rows, cols, -1))
    return img

def Split_Tiles(tiles_list, xsz, ysz, stride=256, patch_size=256):
   
    coor = []
    for i in tiles_list:
        b = np.random.choice([-1, 1])
        if b == 1:
            x = np.arange(0, xsz - patch_size + 1, b*stride)
        else:
            x = np.arange(xsz - patch_size, -1, b*stride)
       
        b = np.random.choice([-1, 1])
        if b == 1:
            y = np.arange(0, ysz - patch_size + 1, b*stride)
        else:
            y = np.arange(ysz - patch_size, -1, b*stride)
       
        coor += list(itertools.product(x + i[0], y + i[1]))

    for i in range(len(coor)):
        coor[i] = (coor[i][0], coor[i][1], i)

    return coor

def Split_Image(obj, rows, cols, no_tiles_h, no_tiles_w, random_tiles = False):

    xsz = rows // no_tiles_h
    ysz = cols // no_tiles_w

    if random_tiles == 'random':

        # Tiles coordinates
        h = np.arange(0, rows, xsz)
        w = np.arange(0, cols, ysz)
        if (rows % no_tiles_h): h = h[:-1]
        if (cols % no_tiles_w): w = w[:-1]
        tiles = list(itertools.product(h, w))

        np.random.seed(3); np.random.shuffle(tiles)

        # Take test tiles
        idx = len(tiles) * 50 // 100; idx += (idx == 0)
        test_tiles = tiles[:idx]
        train_tiles = tiles[idx:]
        # Take validation tiles
        idx = len(train_tiles) * 10 // 100; idx += (idx == 0)
        val_tiles = train_tiles[:idx]
        train_tiles = train_tiles[idx:]

        mask = np.ones((rows, cols))
        for i in val_tiles:
            finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
            finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
            mask[i[0]:finx, i[1]:finy] = .6
        for i in test_tiles:
            finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
            finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
            mask[i[0]:finx, i[1]:finy] = 0
        
        # save_mask = Image.fromarray(np.uint8(mask*255))
        # save_mask.save('../datasets/' + obj.args.dataset_name + '/mask_train_val_test.tif')
        # np.save('../datasets/' + obj.args.dataset_name + '/tiles', tiles)

    elif random_tiles == 'k-fold':

        k = obj.args.k
        mask = Image.open('../datasets/' + obj.args.dataset_name + '/mask_train_val_test_fold_' + str(k) + '.tif')
        mask = np.array(mask) / 255

        # tiles = np.load('../datasets/' + obj.args.dataset_name + '/tiles.npy')

        # # Split in folds
        # size_fold = len(tiles) // obj.args.n_folds
        # test_tiles = tiles[k*size_fold:(k+1)*size_fold]
        # train_tiles = np.concatenate((tiles[:k*size_fold], tiles[(k+1)*size_fold:]))
        # # Take validation tiles
        # np.random.shuffle(train_tiles)
        # idx = len(train_tiles) * 10 // 100; idx += (idx == 0)
        # val_tiles = train_tiles[:idx]
        # train_tiles = train_tiles[idx:]

        # mask = np.ones((rows, cols))
        # for i in val_tiles:
        #     finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
        #     finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
        #     mask[i[0]:finx, i[1]:finy] = .6
        # for i in test_tiles:
        #     finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
        #     finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
        #     mask[i[0]:finx, i[1]:finy] = 0
        
        # save_mask = Image.fromarray(np.uint8(mask*255))
        # save_mask.save('../datasets/' + obj.args.dataset_name + '/mask_train_val_test_fold_' + str(k) + '.tif')

    elif random_tiles == 'fixed':
        # Distribute the tiles from a mask
        mask = Image.open('../datasets/' + obj.args.dataset_name + '/mask_train_val_test.tif')
        mask = np.array(mask) / 255
    
    return mask

def Split_in_Patches(rows, cols, patch_size, mask,
                     lbl, augmentation_list, cloud_mask, 
                     prefix=0, percent=0):

    """
    Everything  in this function is made operating with
    the upper left corner of the patch
    """
    
    # Percent of overlap between consecutive patches.
    overlap = round(patch_size * percent)
    overlap -= overlap % 2
    stride = patch_size - overlap
    # Add Padding to the image to match with the patch size
    step_row = (stride - rows % stride) % stride
    step_col = (stride - cols % stride) % stride
    pad_tuple_msk = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)) )
    mask_pad = np.pad(mask, pad_tuple_msk, mode = 'symmetric')
    lbl = np.pad(lbl, pad_tuple_msk, mode = 'symmetric')
    cloud_mask = np.pad(cloud_mask, pad_tuple_msk, mode = 'symmetric')

    k1, k2 = (rows+step_row)//stride, (cols+step_col)//stride
    print('Total number of patches: %d x %d' %(k1, k2))

    train_mask = np.zeros_like(mask_pad)
    val_mask = np.zeros_like(mask_pad)
    test_mask = np.zeros_like(mask_pad)
    train_mask[mask_pad==1] = 1
    test_mask[mask_pad==0] = 1
    val_mask = (1-train_mask) * (1-test_mask)

    train_patches, val_patches, test_patches = [], [], []
    only_bck_patches = 0
    cloudy_patches = 0
    for i in range(k1):
        for j in range(k2):
            # Train
            if train_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                if cloud_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                    cloudy_patches += 1
                    continue
                for k in augmentation_list:
                    train_patches.append((prefix, i*stride, j*stride, k))
                if not lbl[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                    # train_patches.append((prefix, i*stride, j*stride, 0))
                    only_bck_patches += 1
            # Test                !!!!!Not necessary with high overlap!!!!!!!!
            elif test_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                test_patches.append((prefix, i*stride, j*stride, 0))
            # Val                 !!!!!Not necessary with high overlap!!!!!!!!
            elif val_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                val_patches.append((prefix, i*stride, j*stride, 0))
    print('Training Patches with background only: %d' %(only_bck_patches))
    print('Patches with clouds: %d' %(cloudy_patches))
    
    return train_patches, val_patches, test_patches, step_row, step_col, overlap

def create_dataset_coordinates(obj, prefix = 0):
    
    '''
        Generate Tiles for trn, val and tst
    '''

    patch_size = obj.fine_size

    # Optical image Limits and number of tiles per axis
    # Particularly for this data
    if obj.args.dataset_name == 'Amazonia_Legal':
        lims = np.array([1, 2551, 1, 5121])
    elif obj.args.dataset_name == 'Cerrado_biome':
        lims = np.array([0, 1719, 0, 1442])

    no_tiles_h, no_tiles_w = 5, 5
    rows, cols = lims[1] - lims[0], lims[3] - lims[2]
    mask_opt = Split_Image(obj, rows, cols, no_tiles_h, 
                           no_tiles_w, random_tiles=obj.args.mask)

    # Create output directories
    folder = '../datasets/' + obj.args.dataset_name + '/Norm_params_' + obj.sar_name
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Loading Labels
    lbl = np.load(obj.lbl_path + obj.lbl_name + '.npy')
    lbl = lbl[lims[0]:lims[1],lims[2]:lims[3]]

    # Loading cloud mask
    if obj.args.dataset_name == 'Amazonia_Legal':
        cloud_mask = np.zeros_like(lbl)
    elif obj.args.dataset_name == 'Cerrado_biome':
        cloud_mask = np.load(obj.cloud_path + '/'+ obj.opt_name + '/' + 
                                obj.opt_name + '_cloud_shadow_mask_01.npy')
        cloud_mask = cloud_mask[lims[0]:lims[1],lims[2]:lims[3]]
    
    # Generate Patches for trn, val and tst
    if obj.args.data_augmentation:
        augmentation_list = [-1]                    # A random transformation each epoch
        # augmentation_list = [0, 1, 2, 3, 4, 5]    # All transformation each epoch
    else:
        augmentation_list = [0]                         # Without transformations
    train_patches, val_patches, test_patches, \
    step_row, step_col, overlap = Split_in_Patches(rows, cols, patch_size,
                                                   mask_opt, lbl, augmentation_list,
                                                   cloud_mask, prefix = prefix,
                                                   percent=obj.args.patch_overlap)
    
    print('--------------------')
    print('Training Patches: %d' %(len(train_patches)))
    print('Validation Patches: %d' %(len(val_patches)))
    print('Testing Patches: %d' %(len(test_patches)))

    # Processing SAR Image
    aux = Image.MAX_IMAGE_PIXELS 
    Image.MAX_IMAGE_PIXELS = None               # prevents DecompressionBombWarning
    sar_vh = load_tiff_image(obj.sar_path + obj.sar_name + '/' + obj.sar_name + '_vh.tif').astype('float32')
    sar_vv = load_tiff_image(obj.sar_path + obj.sar_name + '/' + obj.sar_name + '_vv.tif').astype('float32')
    Image.MAX_IMAGE_PIXELS = aux
    sar = np.concatenate((np.expand_dims(sar_vh, 2), np.expand_dims(sar_vv, 2)), axis=2)
    sar = sar[3*lims[0]:3*lims[1], 3*lims[2]:3*lims[3], :]
    sar = 10**(sar/10)                          # convert from dB to linear
    # SAR Normalization
    sar[sar > 1.0] = 1.0                        # Removing outliers
    mask_sar = np.ones((3*rows, 3*cols))
    # mask_sar[sar[:, :, 0] == 1] = 0           # avoid the outliers
    # mask_sar[sar[:, :, 1] == 1] = 0
    # # Only take train region for the calculus of the normalization parameters
    # mask = mask_opt.repeat(3, axis=0).repeat(3, axis=1) # Upsample using nearest neighbour
    # mask_sar = mask_sar * (mask!=0)
    save_mask = Image.fromarray(np.uint8(mask_sar*255))
    save_mask.save(folder + '/norm_mask_sar.tif')
    np.save(obj.sar_path + obj.sar_name + '/' + obj.sar_name, sar)
    sar = Normalization(sar, mask_sar, obj.norm_type, scaler_name=folder + '/sar_' + obj.sar_name)

    # Processing OPT Image
    opt = np.load(obj.opt_path + obj.opt_name + '.npy').astype('float32')
    opt = opt.transpose((1, 2, 0))
    opt = opt[lims[0]:lims[1],lims[2]:lims[3], :]
    # OPT Normaization
    opt[opt > 30000] = 30000                    # Removing outliers
    # Only take train region for the calculus of the normalization parameters
    mask_opt[mask_opt != 0] = 1
    # for i in range(opt.shape[2]):             # avoid the outliers
    #     mask_opt[opt[:,:,i]==30000] = 0
    save_mask = Image.fromarray(np.uint8(mask_opt*255))
    save_mask.save(folder + '/norm_mask_opt.tif')
    if obj.norm_type == 'wise_frame_mean':
        opt = Normalization(opt, mask_opt, 'std', scaler_name=folder + '/opt_' + obj.opt_name) 
    else:
        opt = Normalization(opt, mask_opt, obj.norm_type, scaler_name=folder + '/opt_' + obj.opt_name) 
    
    # Add Padding to the images to match with the patch size
    data_dic = {}
    pad_tuple_opt = ( (overlap//2, overlap//2+step_row), (overlap//2, overlap//2+step_col), (0,0) )
    pad_tuple_lbl = ( (overlap//2, overlap//2+step_row), (overlap//2, overlap//2+step_col) )
    pad_tuple_sar = ( (3*overlap//2, 3*(overlap//2+step_row)), (3*overlap//2, 3*(overlap//2+step_col)), (0,0) )
    data_dic["opt_t"    + str(prefix)] = np.pad(opt, pad_tuple_opt, mode = 'symmetric')
    data_dic["labels_t" + str(prefix)] = np.pad(lbl, pad_tuple_lbl, mode = 'symmetric')
    data_dic["sar_t"    + str(prefix)] = np.pad(sar, pad_tuple_sar, mode = 'symmetric')

    print('Dataset created!!')
    return train_patches, val_patches, test_patches, data_dic

def create_dataset_both_images(obj):
    
    obj.sar_name = obj.sar_name_t0
    obj.opt_name = obj.opt_name_t0
    obj.lbl_name = obj.lbl_name_t0
    train_patches_0, val_patches_0, \
        test_patches_0, data_dic_0 = create_dataset_coordinates(obj, prefix = 0)

    obj.sar_name = obj.sar_name_t1
    obj.opt_name = obj.opt_name_t1
    obj.lbl_name = obj.lbl_name_t1
    obj.args.patch_overlap += 0.03
    train_patches_1, val_patches_1, \
        test_patches_1, data_dic_1 = create_dataset_coordinates(obj, prefix = 1)
    obj.args.patch_overlap -= 0.03

    train_patches = train_patches_0 + train_patches_1
    val_patches   = val_patches_0   + val_patches_1
    test_patches  = test_patches_0  + test_patches_1
    data_dic    = {**data_dic_0, **data_dic_1}

    return train_patches, val_patches, test_patches, data_dic



def Transform(arr, b):

    sufix = ''

    if b == 1:
        arr = np.rot90(arr, k = 1)
        sufix = '_rot90'
    elif b == 2:
        arr = np.rot90(arr, k = 2)
        sufix = '_rot180'
    elif b == 3:
        arr = np.rot90(arr, k = 3)
        sufix = '_rot270'
    elif b == 4:
        arr = np.flipud(arr)
        sufix = '_flipud'
    elif b == 5:
        arr = np.fliplr(arr)
        sufix = '_fliplr'
    elif b == 6:
        if len(arr.shape) == 3:
            arr = np.transpose(arr, (1, 0, 2))
        elif len(arr.shape) == 2:
            arr = np.transpose(arr, (1, 0))
        sufix = '_transpose'
    elif b == 7:
        if len(arr.shape) == 3:
            arr = np.rot90(arr, k = 2)
            arr = np.transpose(arr, (1, 0, 2))
        elif len(arr.shape) == 2:
            arr = np.rot90(arr, k = 2)
            arr = np.transpose(arr, (1, 0))
        sufix = '_transverse'

    return arr, sufix

def Data_augmentation(sar_t0, opt_t0,
                      id_transform,
                      fine_size=256,
                      random_crop_transformation=False,
                      labels=False):

    if id_transform == -1:
        id_transform = np.random.randint(6)
    
    sar_t0, _ = Transform(sar_t0, id_transform)
    opt_t0, _ = Transform(opt_t0, id_transform)
    if labels is not False:
        labels, _ = Transform(labels, id_transform)

    if random_crop_transformation and np.random.rand() > .5:
        dif_size = round(fine_size * 10/100)
        h1 = np.random.randint(dif_size + 1)
        w1 = np.random.randint(dif_size + 1)

        sar_t0 = np.float32(resize(sar_t0, (3*(dif_size+fine_size), 3*(dif_size+fine_size)), preserve_range=True))
        opt_t0 = np.float32(resize(opt_t0, ((dif_size+fine_size), (dif_size+fine_size)), preserve_range=True))

        sar_t0 = sar_t0[3*h1:3*(h1+fine_size), 3*w1:3*(w1+fine_size)]
        opt_t0 = opt_t0[h1:h1+fine_size, w1:w1+fine_size]
        if labels is not False:
            labels = np.float32(resize(labels, ((dif_size+fine_size), (dif_size+fine_size)), preserve_range=True, order=0))
            labels = labels[h1:h1+fine_size, w1:w1+fine_size]
    
    return sar_t0, opt_t0, labels

def Take_patches(patch_list, idx, data_dic,
                 fine_size=256,
                 random_crop_transformation=False,
                 labels=False):

    sar_t0 = data_dic['sar_t' + str(patch_list[idx][0])] \
                     [3*patch_list[idx][1]:3*(patch_list[idx][1]+fine_size),
                      3*patch_list[idx][2]:3*(patch_list[idx][2]+fine_size), :]
    opt_t0 = data_dic['opt_t' + str(patch_list[idx][0])] \
                     [patch_list[idx][1]:patch_list[idx][1]+fine_size,
                      patch_list[idx][2]:patch_list[idx][2]+fine_size, :]
    if labels is not False:
        labels = data_dic['labels_t' + str(patch_list[idx][0])] \
                         [patch_list[idx][1]:patch_list[idx][1]+fine_size,
                          patch_list[idx][2]:patch_list[idx][2]+fine_size]

    sar_t0, opt_t0, labels = Data_augmentation(sar_t0,
                                               opt_t0,
                                               patch_list[idx][3],
                                               fine_size=fine_size,
                                               random_crop_transformation=random_crop_transformation,
                                               labels=labels)

    return sar_t0, opt_t0, labels

def save_samples_multiresolution(self, patch_list, output_path, 
                                 idx=6, epoch=0, labels=False):

    patches = Take_patches(patch_list, idx, data_dic = self.data_dic,
                           fine_size=self.fine_size,
                           random_crop_transformation=False,
                           labels=labels)
    sar_t0 = patches[0][np.newaxis, ...]
    opt_t0_fake = self.sess.run(self.fake_opt_t0_sample,
                                feed_dict={self.sar_t0: sar_t0})

    # # # # VISUALYZING THE PATCHES # # # # # 
    if patch_list[idx][0] == 0:
        self.sar_name = self.sar_name_t0
        self.opt_name = self.opt_name_t0
        self.lbl_name = self.lbl_name_t0
    elif patch_list[idx][0] == 1:
        self.sar_name = self.sar_name_t1
        self.opt_name = self.opt_name_t1
        self.lbl_name = self.lbl_name_t1

    if self.norm_type == 'wise_frame_mean':
        scaler_opt = joblib.load('../datasets/' + self.args.dataset_name + '/Norm_params_' + self.sar_name + \
                                 '/opt_' + self.opt_name + '_' + 'std' + '.pkl')
    else:
        scaler_opt = joblib.load('../datasets/' + self.args.dataset_name + '/Norm_params_' + self.sar_name + \
                                 '/opt_' + self.opt_name + '_' + self.norm_type + '.pkl')
    
    opt_t0_fake = Denormalization(opt_t0_fake[0,:,:,:], scaler_opt)
    opt_t0_fake = opt_t0_fake[:, :, [3, 2, 1]] / (2**16 - 1)
    opt_t0_fake[:, :, 0] = exposure.equalize_adapthist(opt_t0_fake[:, :, 0], clip_limit=0.02)
    opt_t0_fake[:, :, 1] = exposure.equalize_adapthist(opt_t0_fake[:, :, 1], clip_limit=0.02)
    opt_t0_fake[:, :, 2] = exposure.equalize_adapthist(opt_t0_fake[:, :, 2], clip_limit=0.02)
    opt_t0_fake = Image.fromarray(np.uint8(opt_t0_fake*255))
    opt_t0_fake.save(output_path + '/' + str(patch_list[idx][0]) + '_opt_fake_' + str(epoch) + '.tiff')

    if labels is not False:

        scaler_sar = joblib.load('../datasets/' + self.args.dataset_name + '/Norm_params_' + self.sar_name + \
                                 '/sar_' + self.sar_name + '_' + self.norm_type + '.pkl')
        sar_t0 = Denormalization(sar_t0[0,:,:,:], scaler_sar)
        sar_vh = exposure.equalize_adapthist(sar_t0[:,:,0], clip_limit=0.02)
        sar_vh = Image.fromarray(np.uint8(sar_vh*255))
        sar_vh.save(output_path + '/' + str(patch_list[idx][0]) + '_sar_vh.tiff')
        sar_vv = exposure.equalize_adapthist(sar_t0[:,:,1], clip_limit=0.02)
        sar_vv = Image.fromarray(np.uint8(sar_vv*255))
        sar_vv.save(output_path + '/' + str(patch_list[idx][0]) + '_sar_vv.tiff')

        opt_t0 = patches[1]
        opt_t0 = Denormalization(opt_t0, scaler_opt)
        opt_t0 = opt_t0[:, :, [3, 2, 1]] / (2**16 - 1)
        opt_t0[:, :, 0] = exposure.equalize_adapthist(opt_t0[:, :, 0], clip_limit=0.02)
        opt_t0[:, :, 1] = exposure.equalize_adapthist(opt_t0[:, :, 1], clip_limit=0.02)
        opt_t0[:, :, 2] = exposure.equalize_adapthist(opt_t0[:, :, 2], clip_limit=0.02)
        opt_t0 = Image.fromarray(np.uint8(opt_t0*255))
        opt_t0.save(output_path + '/' + str(patch_list[idx][0]) + '_opt_real.tiff')

        labels = patches[2]
        labels = Image.fromarray(np.uint8(labels*255))
        labels.save(output_path + '/' + str(patch_list[idx][0]) + '_labels.tiff')




