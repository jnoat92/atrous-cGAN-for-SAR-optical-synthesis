#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:53:42 2020

@author: daliana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:01:12 2020

@author: daliana
"""
from keras.optimizers import Adam
import sys
from PIL import Image
import numpy as np
from time import sleep
import multiprocessing
import itertools
from sklearn.utils import shuffle
import math
import keras
import statistics
from sklearn import preprocessing as pre
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
from skimage.morphology import binary_dilation, disk, binary_erosion, area_opening
from reconstruction import Image_reconstruction, rgb2gray, gray2rgb
from sklearn.metrics import confusion_matrix
import os
#

def Directory():

    dir = '../../test_images/Cerrado_biome/k-fold/'
    dir_Images_past = dir + '2/atrous_bs1_wise_frame_mean_ps128_op83_ep100_fold_' + str(kk) + '_2017_08.npz'
    dir_Image_present = dir + '2/atrous_bs1_wise_frame_mean_ps128_op83_ep100_fold_' + str(kk) + '_2018_08.npz'
    
    # dir = '../../../datasets/Cerrado_biome/Landsat_8/Images/'
    # dir_Images_past = dir + '18_08_2017_image.npy'
    # dir_Image_present = dir + '21_08_2018_image.npy'

    dir = '../../../datasets/Cerrado_biome/Landsat_8/Reference/'
    dir_past_reference = dir + 'PAST_REFERENCE_FOR_2018_EPSG4674.npy'
    dir_reference = dir + "REFERENCE_2018_EPSG4674.npy"
       
    dataset = dir_Images_past, dir_Image_present, dir_reference, dir_past_reference

    return dataset

#%%
def load_im(): 

    dataset = Directory()

    im = []
    for j in dataset:
        im_ = np.load(j)
        if j[-4 :] == ".npz" :
            im_ = im_['arr_0']
        else:
            if len(im_.shape) == 3:
                im_ = im_.transpose((1, 2, 0))
        im.append([im_, j])

    x = []
    files_name = []
        
    for i in range(len(im)):
        x_set, z = im[i]
        x.append(x_set)
        files_name.append(z)

    return x, files_name
#%%
def NDVI_band(image):
# This is to pick up the band 8    
    band_8 = (image[:,:,4] - image[:,:,3])/(image[:,:,4] + image[:,:,3])
    band_8 = np.expand_dims(band_8, axis = 2)
    return band_8
#%% 
def split_tail(rows, cols, no_tiles_h, no_tiles_w):   
   
    h = np.arange(0, rows, int(rows/no_tiles_h))
    w = np.arange(0, cols, int(cols/no_tiles_w))
   
    #Tiles coordinates
    tiles = list(itertools.product(h,w))
    
    return tiles   

#%%   
def Normalization(im):
   
######### Normalization ######################
    rows, cols, c = im.shape 
    im = im.reshape((rows * cols, c))  
#    scaler = pre.MinMaxScaler((-1,1)).fit(im)
    if Arq == 3:
        scaler = pre.StandardScaler(with_std=False).fit(im)
    else:
        scaler = pre.StandardScaler().fit(im)
    Norm_Image = np.float32(scaler.transform(im))
    Norm_Image = Norm_Image.reshape((rows, cols, c))
    return  Norm_Image
##############################################
    
#%% 
def Hot_encoding(ref):    
######## Hot encoding #########################
    rows, cols = ref.shape 
    classes = len(np.unique(ref))
    imgs_mask_cat = ref.reshape(-1)
    imgs_mask_cat = keras.utils.to_categorical(imgs_mask_cat, classes)
    gdt = imgs_mask_cat.reshape(rows, cols, classes) 
    return  gdt
###############################################  

#%%
def Using_buffer(ref):
    
    selem = disk(4)
    ero = disk(2)
    erosion = np.uint(binary_erosion(ref, ero)) 
    dilation = np.uint(binary_dilation(ref, selem)) 
    buffer  = dilation - erosion
    ref[buffer == 1] = 2
    return ref

#%%
def create_mask():
    mask = np.ones((rows, cols))
    for i in val_set:
        mask[i[0]:i[0] + int(rows/no_tiles_h), i[1]:i[1] + int(cols/no_tiles_w)] = .5
    for i in test_set:
        mask[i[0]:i[0] + int(rows/no_tiles_h), i[1]:i[1] + int(cols/no_tiles_w)] = 0
    return mask

#%%
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
        arr = np.transpose(arr, (1, 0, 2))
        sufix = '_transpose'
    elif b == 7:
        arr = np.rot90(arr, k = 2)
        arr = np.transpose(arr, (1, 0, 2))
        sufix = '_transverse'

    return arr
#%%
def Add_padding(reference):
    # Add Padding to the image to match with the patch size
    pad_tuple_msk = ( (overlap//2, overlap//2 + step_row), (overlap//2, overlap//2 + step_col) )
    pad_tuple_img = ( (overlap//2, overlap//2 + step_row), (overlap//2, overlap//2 + step_col), (0, 0) )
    
    mask_pad = np.pad(mask, pad_tuple_msk, mode = 'symmetric')    
    img_pad = np.pad(Norm_image, pad_tuple_img, mode = 'symmetric')
    gdt_pad = np.pad(reference, pad_tuple_msk, mode = 'symmetric')
    
    return mask_pad, img_pad, gdt_pad

#%%
def split_patches(k1, k2):
    
    for i in range(k1):
        for j in range(k2):
            # Test
            if test_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                test_patches.append((i*stride, j*stride))
            elif val_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                # We only do data augmentation to the patches where there is a positive sample.
                if gdt_pad[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
#                    Only patches with samples
                    for q in data_augmentation_index:                   
                        val_patches.append((i*stride, j*stride, q))
                else:
                    val_patches.append((i*stride, j*stride, 0))

            elif train_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                # We only do data augmentation to the patches where there is a positive sample.
                if gdt_pad[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
#                    Only patches with samples
                    for q in data_augmentation_index:                   
                        train_patches.append((i*stride, j*stride, q))
                else:
                    train_patches.append((i*stride, j*stride, 0))
                    
    return train_patches, val_patches, test_patches  

#%%
def patch_image(img_pad, gdt, data_batch):

    patch_im = []
    patch_gdt = []    

#   Loading the patches in the image
    for j in range(len(data_batch)):
        I_patch = img_pad[data_batch[j][0]: data_batch[j][0] + patch_size, data_batch[j][1]: data_batch[j][1] + patch_size,:]
#        Apply transformations to the image patches. 
        I_patch = Transform (I_patch, data_batch[j][2])  
        patch_im.append(I_patch)  
        
        gdt_patch = gdt[data_batch[j][0]: data_batch[j][0] + patch_size, data_batch[j][1]: data_batch[j][1] + patch_size, :]
#        Apply transformations to the reference patches. 
        gdt_patch = Transform (gdt_patch, data_batch[j][2]) 
        patch_gdt.append(gdt_patch) 
    
    patch_im = np.array(patch_im)

    patch_gdt = np.array(patch_gdt)

    return patch_im, patch_gdt

#%%
def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
        
        weights = K.variable(weights)
            
        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss
        return loss

#%%
#%%
def network(Arq, reference, weights, patch_size):
       
    from Arquitecturas.U_net import Unet
    from Arquitecturas.segnet_unpooling import Segnet
    # from Arquitecturas.deeplabv3p import Deeplabv3p
    # from Arquitecturas.DenseNet import Tiramisu
#    from Arquitecturas.fusion import Tiramisu
    
    opt = Adam(lr=lr)
    
    if Arq == 1:
        print("Start Segnet")
        net = Segnet(nClasses = len(np.unique(reference)), optimizer = None, input_width = patch_size , input_height = patch_size , nChannels = 16)

    elif Arq == 2:
        print("Start Unet")
        net = Unet(len(np.unique(reference)), patch_size, patch_size , 16) 

    elif Arq ==3:
        print("Start DeepLabv3p")
        net = Deeplabv3p(input_tensor=None, infer = False,
                input_shape=(patch_size, patch_size, 16), classes= len(np.unique(reference)), backbone='mobilenetv2', OS=16, alpha=1.)
    else:           
        print("Start DenseNet")
        net = Tiramisu(input_shape = (patch_size,patch_size,16), n_classes = len(np.unique(reference)), n_filters_first_conv = 32, 
                      n_pool = 3, growth_rate = 8, n_layers_per_block = [4,4,4,4,4,4,4,4,4,4,4],  dropout_p = 0)
    
    weighted_categorical = weighted_categorical_crossentropy(weights)
    
    net.compile(loss = weighted_categorical, optimizer = opt, metrics=["accuracy"])
      
    return net
        
#%%
def train_test(img_pad, gdt, dataset, flag):   

     global net  
     n_batch = len(dataset) // batch_size

     loss = np.zeros((1 , 2)) 

########### Training per batch ####################
     for i in range(n_batch):
#          Data_batch is going to be the shape 
#         (x_coordinates, y coordinates, transformation_index) in the batch
        data_batch = dataset[i * batch_size : (i + 1) * batch_size]

        patch_im, patch_gdt = patch_image(img_pad, gdt, data_batch)

        if flag:
            loss += net.train_on_batch(patch_im, patch_gdt)
        else:
            loss += net.test_on_batch(patch_im, patch_gdt)
            
     if len(dataset) % batch_size:        
        
         data_batch = dataset[n_batch * batch_size : ] 
         
         patch_im, patch_gdt = patch_image(img_pad, gdt, data_batch)
         
         if flag:
            loss += net.train_on_batch(patch_im, patch_gdt)
         else:
            loss += net.test_on_batch(patch_im, patch_gdt)
        # Here, we have a remanent batch, so we have to add 1 to the n_batch    
         loss= loss/(n_batch + 1)
     else:
         loss= loss/n_batch

     return loss

#%%
def Train(img_pad, gdt, train_patches, val_patches):

    global net, Arq  
    net = network(Arq, reference, weights, patch_size)  
    
    loss_train_plot = []
    accuracy_train = []
    
    loss_val_plot = []
    accuracy_val = []
    
    patience_cnt = 0
    minimum = 10000.0
              
    print('Start the training')
    start = time.time()

    for epoch in range(epochs):
        
        loss_train = np.zeros((1 , 2))
        loss_val = np.zeros((1 , 2))
        
        # Shuffling the train data 
        train_patches = shuffle(train_patches, random_state = 0)

        # Evaluating the network in the train set
        loss_train = train_test(img_pad, gdt, train_patches, flag = 1) 
        
#       To see the training curves
        loss_train_plot.append(loss_train[0, 0])
        accuracy_train.append(100 * loss_train[0, 1]) 
        
        print("%d [training loss: %f, Train Acc: %.2f%%]" %(epoch, loss_train[0, 0], 100 * loss_train[0, 1]))
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
           
         ################################################
        # Evaluating the network in the validation set
        
        loss_val = train_test(img_pad, gdt, val_patches, flag = 0) 
        
        print("%d [Validation loss: %f, Validation Acc: %.2f%%]" %(epoch , loss_val[0 , 0], 100 * loss_val[0 , 1]))
#       To see the validation curves
        loss_val_plot.append(loss_val[0, 0])
        accuracy_val.append(100 * loss_val[0, 1])        

#        Performing Early stopping
        if  loss_val[0,0] < minimum:
          patience_cnt = 0
          minimum = loss_val[0,0]
          
#          Saving the best model for all runs.
          if Arq == 1:
              net.save(fold_dir + '/best_model_Segnet_%d.h5'%(k))
          elif Arq == 2 :
              net.save(fold_dir + '/best_model_Unet_%d.h5'%(k))
          elif Arq == 3 :
              net.save(fold_dir + '/best_model_Deep_%d.h5'%(k))
          else:
              net.save(fold_dir + '/best_model_Dense_%d.h5'%(k))
              
        else:
          patience_cnt += 1
#    
        if patience_cnt > 10:
          print("early stopping...")
          break
    
    del net
    
    return loss_train_plot, accuracy_train, loss_val_plot, accuracy_val

#%%
if __name__=='__main__':
    
    arq = [1, 2]

    for Arq in arq:

        batch_size = 16
        lr = 0.001
        weights = [0.4, 2] 
        epochs = 100

        P =  0.8
        no_tiles_h, no_tiles_w = 5, 5
        patch_size = 128
        overlap = round(patch_size * P)
        overlap -= overlap % 2
        stride = patch_size - overlap

        cancel_past_ref = 1
        cancel_buffer = 1

        for kk in range(5):
            fold_dir = './fold_' + str(kk)
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            mask = Image.open('../../../datasets/Cerrado_biome/mask_train_val_test_fold_' + str(kk) + '.tif')
            mask = np.array(mask) / 255

            accuracy = []
            f1_score = []
            precision = []
            recall = []
            IoU = []
            true_negative = []
            false_positive = []
            false_negative = []
            true_positive = []
                
            x, files_name = load_im()

            for i in range(len(x)):
                # Concatenating the band_8 with the others 7 bands
                if i <= 1:
                    band = NDVI_band(x[i]) 
                    x[i] = np.concatenate((x[i],band), axis = 2)
                
            I = np.concatenate((x[0], x[1]), axis = 2) 
            rows, cols, c = I.shape
            Norm_image = Normalization(I) 

            past_reference = x[3]
            act_reference = x[2] 

            reference = act_reference.copy()

            if cancel_buffer:
                reference = Using_buffer(reference)
                if len(weights) == 2:
                    weights.append(0)
                print('Cancel Buffer')
            
            if cancel_past_ref:
                reference[past_reference == 1] = 2
                if len(weights) == 2:
                    weights.append(0)
                print('Cancel Past Reference')

            tiles_Image = split_tail(rows, cols, no_tiles_h, no_tiles_w)

            N_Run = 5

            for k in range(N_Run):

                print("_____Run number %d---%d_______" %(kk, k))

        #        tiles_Image = shuffle(tiles_Image, random_state = k)
        #        
        #        train_set = tiles_Image[: math.ceil(0.25 * len(tiles_Image))]
        #        val_set = train_set[: math.ceil(0.05 * len(train_set))]
        #        train_set = train_set[math.ceil(0.05 * len(train_set)):]
        #        test_set = tiles_Image[math.ceil(0.25 * len(tiles_Image)) :]
        #           
        #        mask = create_mask()
                
        #        mask = Image.open('mask_train_val_test.tif')
                
        #        This is the number of pixels you have to add in the padding step
                step_row = (stride - rows % stride) % stride
                step_col = (stride - cols % stride) % stride
                
                mask_pad, img_pad, gdt_pad = Add_padding(reference)
                
        #        plt.figure()
        #        imgplot = plt.imshow(train_mask)
        #        plt.show()    
        #        
                # List for data augmentation.
                data_augmentation_index = [0, 1, 4, 5]
                
                train_patches, val_patches, test_patches = [], [], []
                k1, k2 = (rows + step_row)//stride, (cols + step_col)//stride
                print('Total number of patches: %d x %d' %(k1, k2))
                
        #        Create the mask to train, validation and test.
                train_mask = np.zeros_like(mask_pad)
                val_mask = np.zeros_like(mask_pad)
                test_mask = np.zeros_like(mask_pad)
                train_mask[mask_pad == 1] = 1
                test_mask[mask_pad == 0] = 1
                val_mask = (1 - train_mask) * (1 - test_mask)
                
                # Split paches index            
                train_patches, val_patches, test_patches = split_patches(k1, k2)

                # Train patches
                print('Train patches: %d' %(len(train_patches)))
                print('Val patches: %d' %(len(val_patches)))
                print('Test patches: %d' %(len(test_patches)))

                gdt = Hot_encoding(gdt_pad)  

                loss_train_plot, accuracy_train, loss_val_plot, accuracy_val = Train(img_pad, gdt, train_patches, val_patches)
            
                #%% 
        ##############################Testing the models.######################################
                net = network(Arq, reference, weights, patch_size)  
                if Arq == 1:
                    net.load_weights(fold_dir + '/best_model_Segnet_%d.h5'%(k))  
                elif Arq == 2:
                    net.load_weights(fold_dir + '/best_model_Unet_%d.h5'%(k))
                elif Arq == 3:
                    net.load_weights(fold_dir + '/best_model_Deep_%d.h5'%(k))
                else:
                    net.load_weights(fold_dir + '/best_model_Dense_%d.h5'%(k))

                obj = Image_reconstruction(net, 3, patch_size = patch_size, overlap_percent = P)
                #        Prediction stage
                predict_probs = obj.Inference(Norm_image)

                predict_labels = predict_probs.argmax(axis=-1)
            
                test_mask = np.zeros_like(mask)
                test_mask[mask == 0] = 1
                
        #        plt.figure()
        #        imgplot = plt.imshow(test_mask)
        #        plt.show()  
                
                print('Predicted labels:')
                predict_labels[predict_labels == 2] = 0
                test_mask[reference == 2] = 0 
                        
                less_6ha_predict = predict_labels - area_opening(predict_labels.astype('int'),
                                                area_threshold = 69, connectivity=1)
                test_mask[less_6ha_predict == 1] = 0
                print(np.unique(less_6ha_predict)) 

                ######## Only to see the image the test part
                predict_total = predict_labels * test_mask
                test_reference = act_reference * test_mask
                predict_see = gray2rgb(np.uint(predict_total))
                reference_see = gray2rgb(np.uint(test_reference))

                cv2.imwrite(fold_dir + '/predict_total_'+ str(k) + '.tiff', predict_see)
                cv2.imwrite(fold_dir + '/test_reference_'+ str(k) + '.tiff', reference_see)
                test_mask_see = gray2rgb(np.uint(test_mask))
                cv2.imwrite(fold_dir + '/test_mask_'+ str(k) + '.tiff', test_mask_see)
                ###################################################

                y_pred = predict_labels[test_mask == 1]
                y_true = act_reference[test_mask == 1]   
                    
        #        ____Positive Class Metrics____
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

                accu = (tp + tn)/(tn + fp + fn + tp)
                Prec = tp/(tp + fp)
                R = tp/(tp + fn)
                F1 = 2 * Prec* R/(Prec + R)
                
                intersection = np.sum(np.logical_and(y_pred, y_true)) # Logical AND  
                union = np.sum(np.logical_or(y_pred, y_true)) # Logical OR 
                Iou = intersection/union

                print('Positive class')
                print(tn, fp, fn, tp)
                print('Test accuracy:%.2f' %(100*accu))
                print('Intersection over Union:%.2f' %(100*Iou))
                print('Test f1score:%.2f' %(100*F1))
                print('Test precision:%.2f' %(100*Prec))
                print('Test recall:%.2f' %(100*R))
                    
                lt = 'a' 
                if not k:
                    lt = 'w'
                if Arq == 1:
                    file_metrics = open(fold_dir + "/metrics_Segnet.txt", lt)
                elif Arq == 2:
                    file_metrics = open(fold_dir + "/metrics_Unet.txt", lt)
                elif Arq == 3:
                    file_metrics = open(fold_dir + "/metrics_DeepLab.txt", lt)
                else:
                    file_metrics = open(fold_dir + "/metrics_FCDenseNet.txt", lt)
                
                if not k:
                    file_metrics.write('_ _ _  _K-Fold: %d _ _ _\n'%(kk))
                file_metrics.write('_ _ __ _ __ _ __ _ __ _ _\n')
                file_metrics.write('_ _ _ Run: %d\n'%(k))
                file_metrics.write('Overall Accuracy: %2f\n'%(100*accu))
                file_metrics.write('IoU: %2f\n\n'%(100*Iou))
                # file_metrics.write('Confusion_matrix:\n')
                # file_metrics.write('TP: %2f\n'%(tp))
                # file_metrics.write('TN: %2f\n'%(tn))
                # file_metrics.write('FP: %2f\n'%(fp))
                # file_metrics.write('FN: %2f\n\n'%(fn))

                file_metrics.write('____Positive class____\n')
                file_metrics.write('F1: %2f\n'%(100*F1))
                file_metrics.write('Precision: %2f\n'%(100*Prec))
                file_metrics.write('Recall: %2f\n\n'%(100*R))


        #        ____Average Metrics____
                tn, fp, fn, tp = confusion_matrix(1-y_true, 1-y_pred).ravel()
                Prec_neg_class = tp/(tp + fp)
                R_neg_class = tp/(tp + fn)
                F1_neg_class = 2 * Prec_neg_class * R_neg_class / (Prec_neg_class + R_neg_class)
                
                Prec = (Prec + Prec_neg_class) / 2
                R = (R + R_neg_class) / 2
                F1 = (F1 + F1_neg_class) / 2

                file_metrics.write('____Average Metrics____\n')
                file_metrics.write('F1: %2f\n'%(100*F1))
                file_metrics.write('Precision: %2f\n'%(100*Prec))
                file_metrics.write('Recall: %2f\n\n\n'%(100*R))

                print('Average Metrics')
                print('Average f1score:%.2f' %(100*F1))
                print('Average precision:%.2f' %(100*Prec))
                print('Average recall:%.2f' %(100*R))
                
                file_metrics.close()

                del net
        #        accuracy.append(accu)
        #        f1_score.append(F1)
        #        precision.append(Prec)
        #        recall.append(R)
        #        IoU.append(Iou)
        #        
        #        true_negative.append(tn)
        #        false_positive.append(fp)
        #        false_negative.append(fn)
        #        true_positive.append(tp)