#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:53:42 2020
/media/javier/LVC_05_DADOS/Javier/GitHub/SAR2Optical/Campo Verde/codes/Classification/FCN_SegNet
No@
"""

import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn import preprocessing as pre
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import keras
from keras import backend as K
from keras.optimizers import Adam
import time
import os
import glob
from reconstruction import Image_reconstruction, rgb2gray, gray2rgb

def load_landsat(path):
    images = sorted(glob.glob(path + '*.tif'))
    band = np.array(Image.open(images[0]))
    rows, cols = band.shape
    img = np.zeros((rows, cols, 7), dtype='float32')
    num_band = 0
    for im in images:
        if 'B8' not in im and 'QB' not in im:
            band = np.array(Image.open(im))
            img[:, :, num_band] = band
            num_band += 1
        if 'QB' in im:
            cloud_mask = np.array(Image.open(im))
            cloud_mask[cloud_mask != 0] = 1
    return img, cloud_mask

#%%
def NDVI_band(image):

    ndvi = (image[:,:,4] - image[:,:,3]) / (image[:,:,4] + image[:,:,3])
    return ndvi[..., np.newaxis]

#%%
def load_im(exp): 

    '''
    Experiments: 1 to 4
    1 - Original optical image
    2 - Fake atrous
    3 - Fake pix2pix
    4 - sar
    '''

    sar_root_path = '../../../datasets/Sentinel_1A/npy_format/'
    opt_root_path = '../../../datasets/LANDSAT/'
    labels_root_path = '../../../datasets/Labels_uint8/'

    sar_img_name = '10_08May_2016.npy'
    opt_img_name = '20160505'
    labels_name = '10_May_2016.tif'

    # MASKS
    mask_train_test = np.load('../../../datasets/Maks/mask_gan_original.npy')
    mask_train_test[mask_train_test != 0] = 1
    mask_train_test = 1 - mask_train_test

    mask_nan = np.load('../../../datasets/Maks/mask_nan.npy')

    mask = np.zeros(mask_train_test.shape)
    mask[mask_train_test != 0] = 1
    mask[mask_nan != 1] = 0.3
    save_mask = Image.fromarray(np.uint8(mask*255))
    save_mask.save('mask.tif')

    # Labels
    labels = np.array(Image.open(labels_root_path + labels_name))
    # # Percent of classes
    # val, counts = np.unique(labels, return_counts=True)
    # print(val); print(counts)
    # print(100*counts[1:] // (np.sum(counts[1:])))

    if exp == 1:
        # Original
        I, _ = load_landsat(opt_root_path + opt_img_name + '/')
        scaler = joblib.load('../../../datasets/Campo_Verde_Crops/opt_may2016_scaler.pkl')
        num_rows, num_cols, bands = I.shape
        I = I.reshape((num_rows * num_cols, bands))
        I = np.float32(scaler.transform(I))
        I = I.reshape((num_rows, num_cols, bands))
        I = np.concatenate((I, NDVI_band(I)), axis=2)

        mask = np.array(Image.fromarray(mask).resize((I.shape[1], I.shape[0]), Image.NEAREST))
        labels = np.array(Image.fromarray(labels).resize((I.shape[1], I.shape[0]), Image.NEAREST))

    elif exp == 2:
        # Fake atrous
        fake_name = '../../test_images/deeplab_crops_monotemp.npz'
        I = np.load(fake_name)
        if fake_name[-4:] == '.npz':
            I = I['arr_0']
        I = np.concatenate((I, NDVI_band(I)), axis=2)

        mask = np.array(Image.fromarray(mask).resize((I.shape[1], I.shape[0]), Image.NEAREST))
        labels = np.array(Image.fromarray(labels).resize((I.shape[1], I.shape[0]), Image.NEAREST))

    elif exp == 3:
        # Fake pix2pix
        fake_name = '../../test_images/unet_papper.npy'
        I = np.load(fake_name)
        if fake_name[-4:] == '.npz':
            I = I['arr_0']
        I = np.concatenate((I, NDVI_band(I)), axis=2)

        mask = np.array(Image.fromarray(mask).resize((I.shape[1], I.shape[0]), Image.NEAREST))
        labels = np.array(Image.fromarray(labels).resize((I.shape[1], I.shape[0]), Image.NEAREST))

    elif exp == 4:
        # SAR
        I = np.load(sar_root_path + sar_img_name).astype('float32')
        scaler = joblib.load('../../../datasets/Campo_Verde_Crops/sar_may2016_10m_scaler.pkl')
        num_rows, num_cols, bands = I.shape
        I = I.reshape((num_rows * num_cols, bands))
        I = np.float32(scaler.transform(I))
        I = I.reshape((num_rows, num_cols, bands))

    # Merging minority classes
    for i in range(12):
        if (i != 0) and (i != 2) and (i != 3):
            labels[labels == i] = 4

    # Percent of classes
    labels_AreaOfInterest = labels[mask != 0.6]
    val, counts = np.unique(labels_AreaOfInterest, return_counts=True)
    print('percent of positive classes:')
    print(val[1:])
    print(100*counts[1:] / (np.sum(counts[1:])))
    print('percent of all classes:')
    print(val)
    perc_labels = 100*counts / (np.sum(counts))
    print(perc_labels)

    # Changing Labels indices for one hot encoding
    j = 0
    for i in val:
        labels[labels == i] = j; j += 1
    
    return I, labels, mask

#%%
def Split_in_Patches(rows, cols, patch_size, mask, labels,
         augmentation_list, percent=0):

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
    labels_pad = np.pad(labels, pad_tuple_msk, mode = 'symmetric')

    k1, k2 = (rows+step_row)//stride, (cols+step_col)//stride
    print('Total number of patches: %d x %d' %(k1, k2))

    train_mask = np.zeros_like(mask_pad)
    test_mask = np.zeros_like(mask_pad)

    train_mask[mask_pad==1] = 1
    test_mask[mask_pad==0] = 1

    train_patches, test_patches = [], []
    for i in range(k1):
        for j in range(k2):
            if labels_pad[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                # Train
                if train_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                    for k in augmentation_list:
                        train_patches.append((i*stride, j*stride, k))
                # Test    !!!!!Not necessary with high overlap!!!!!!!!
                elif test_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                    id_transform = np.random.randint(1, 6)
                    test_patches.append((i*stride, j*stride, id_transform))
    
    return train_patches, test_patches, step_row, step_col, overlap

#%% 
def Hot_encoding(labels):    
    ######## Hot encoding #########################
    rows, cols = labels.shape 
    classes = len(np.unique(labels))
    imgs_mask_cat = labels.reshape(-1)
    imgs_mask_cat = keras.utils.to_categorical(imgs_mask_cat, classes)
    y_true = imgs_mask_cat.reshape(rows, cols, classes) 
    
    return  y_true

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
def network(Arq, num_classes, weights, patch_size, input_chanels):

    from Arquitecturas.U_net import Unet
    from Arquitecturas.segnet_unpooling import Segnet
    # from Arquitecturas.deeplabv3p import Deeplabv3p
    # from Arquitecturas.DenseNet import Tiramisu
    # from Arquitecturas.fusion import Tiramisu
    
    opt = Adam(lr = lr)
    
    if Arq == 1:
        print("Start Segnet")
        net = Segnet(nClasses = num_classes, optimizer = None, input_width = patch_size , input_height = patch_size , nChannels = input_chanels)

    elif Arq == 2:
        print("Start Unet")
        net = Unet(num_classes, patch_size, patch_size , input_chanels) 

    elif Arq ==3:
        print("Start DeepLabv3p")
        net = Deeplabv3p(input_tensor=None, infer = False,
    input_shape=(patch_size, patch_size, input_chanels), classes = num_classes, backbone='mobilenetv2', OS=16, alpha=1.)
    else:           
        print("Start DenseNet")
        net = Tiramisu(input_shape = (patch_size, patch_size, input_chanels), n_classes = num_classes, n_filters_first_conv = 32, 
          n_pool = 3, growth_rate = 8, n_layers_per_block = [4,4,4,4,4,4,4,4,4,4,4],  dropout_p = 0)
    
    weighted_categorical = weighted_categorical_crossentropy(weights)
    
    net.compile(loss = weighted_categorical, optimizer = opt, metrics=["accuracy"])
      
    return net

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
def patch_image(img_pad, gdt, data_batch):

    patch_im = []
    patch_gdt = []    

    # Loading the patches in the image
    for j in range(len(data_batch)):
        I_patch = img_pad[data_batch[j][0]: data_batch[j][0] + patch_size, data_batch[j][1]: data_batch[j][1] + patch_size,:]
        # Apply transformations to the image patches. 
        I_patch = Transform (I_patch, data_batch[j][2])  
        patch_im.append(I_patch)  
        
        gdt_patch = gdt[data_batch[j][0]: data_batch[j][0] + patch_size, data_batch[j][1]: data_batch[j][1] + patch_size, :]
        # Apply transformations to the reference patches. 
        gdt_patch = Transform (gdt_patch, data_batch[j][2]) 
        patch_gdt.append(gdt_patch) 
    
    patch_im = np.array(patch_im)

    patch_gdt = np.array(patch_gdt)

    return patch_im, patch_gdt

#%%
def train_test(img_pad, gdt, dataset, flag):   

    global net  
    n_batch = len(dataset) // batch_size

    loss = np.zeros((1 , 2)) 

    ########### Training per batch ####################
    for i in range(n_batch):
        # Data_batch is going to be the shape 
        # (x_coordinates, y coordinates, transformation_index) in the batch
        data_batch = dataset[i * batch_size : (i + 1) * batch_size]

        patch_im, patch_gdt = patch_image(img_pad, gdt, data_batch)

        if flag:
            loss += net.train_on_batch(patch_im, patch_gdt)
        else:
            loss += net.test_on_batch(patch_im, patch_gdt)
        
    # Remanent batch
    if len(dataset) % batch_size:        
    
        data_batch = dataset[n_batch * batch_size : ] 
        
        patch_im, patch_gdt = patch_image(img_pad, gdt, data_batch)
        
        if flag:
            loss += net.train_on_batch(patch_im, patch_gdt)
        else:
            loss += net.test_on_batch(patch_im, patch_gdt)
        
        loss = loss/(n_batch + 1)
    else:
        loss= loss/n_batch

    return loss

#%%
def Train(img_pad, gdt, train_patches, val_patches):

    global net

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
        
        # To see the training curves
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
        # To see the validation curves
        loss_val_plot.append(loss_val[0, 0])
        accuracy_val.append(100 * loss_val[0, 1])        

        # Performing Early stopping
        if  loss_val[0,0] < minimum:
            patience_cnt = 0
            minimum = loss_val[0,0]
            
            # Saving the best model for all runs.
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

        if patience_cnt > 10:
          print("early stopping at epoch %d" %(epoch))
          break
    
    del net
    
    return loss_train_plot, accuracy_train, loss_val_plot, accuracy_val

#%%
if __name__=='__main__':

    # Params
    Exp = [1, 2, 3, 4]
    arq = [2]

    for exp in Exp:
        for Arq in arq:

            N_Run = 5
            overlap_percent =  0.85
            if exp == 4: overlap_percent =  0.65
            patch_size = 128
            augmentation_list = [0, 1, 4, 5]
            # augmentation_list = [0]

            batch_size = 16
            lr = 0.0007
            scale = 10
            epochs = 1000

            fold_dir = './Campo verde results____experiment(%d)' %(exp)
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            # Load data
            I, labels, mask = load_im(exp)
            rows, cols, channels = I.shape

            # Split Patches
            train_patches, test_patches, \
            step_row, step_col, overlap = Split_in_Patches(rows, cols, patch_size, mask, labels,
                    augmentation_list, percent=overlap_percent)
            
            # idx = len(train_patches) * 10 // 100; idx += (idx==0)
            # val_patches = train_patches[:idx//2] + train_patches[-idx//2:]
            # train_patches = train_patches[idx//2:-idx//2]
            idx = len(test_patches) * 10 // 100; idx += (idx==0)
            val_patches = test_patches[:idx//2] + test_patches[-idx//2:]
            test_patches = test_patches[idx//2:-idx//2]

            print('Train patches: %d' %(len(train_patches)))
            print('Val patches: %d' %(len(val_patches)))
            print('Test patches: %d' %(len(test_patches)))

            # Add Padding
            pad_tuple_img = ( (overlap//2, overlap//2+step_row), (overlap//2, overlap//2+step_col), (0,0) )
            pad_tuple_lbl = ( (overlap//2, overlap//2+step_row), (overlap//2, overlap//2+step_col) )
            I_pad = np.pad(I, pad_tuple_img, mode = 'symmetric')
            labels_pad = np.pad(labels, pad_tuple_lbl, mode = 'symmetric')

            # Percent of classes Training region
            train_mask_aux = np.zeros_like(labels_pad)
            for i in train_patches:
                train_mask_aux[i[0]:i[0]+patch_size, i[1]:i[1]+patch_size] = 1

            labels_AreaOfTraining = labels_pad[train_mask_aux==1]
            val, counts = np.unique(labels_AreaOfTraining, return_counts=True)
            print('percent of positive classes training region:')
            perc_labels = 100*counts[1:] / (np.sum(counts[1:]))
            print(val[1:])
            print(perc_labels)
            # Classes weights
            weights = scale - (scale * perc_labels / 100)
            weights = np.concatenate(([0.0], weights), axis=0)
            print('weights: ', weights)

            # Hot_encoding
            labels_one_hot = Hot_encoding(labels_pad)

            for k in range(N_Run):

                print('_ _ _ Run: %d___lr(%f)   batch_size(%d)   scale(%d)\n'%(k, lr, batch_size, scale))
            
                global net
                net = network(Arq, len(np.unique(labels)), weights, patch_size, channels)
                # net.summary()

                loss_train_plot, accuracy_train, loss_val_plot, accuracy_val = Train(I_pad, labels_one_hot, train_patches, val_patches)

#%% 
                ##############################Testing the models.######################################
                net = network(Arq, len(np.unique(labels)), weights, patch_size, channels)
                if Arq == 1:
                    net.load_weights(fold_dir + '/best_model_Segnet_%d.h5'%(k))  
                elif Arq == 2:
                    net.load_weights(fold_dir + '/best_model_Unet_%d.h5'%(k))
                elif Arq == 3:
                    net.load_weights(fold_dir + '/best_model_Deep_%d.h5'%(k))
                else:
                    net.load_weights(fold_dir + '/best_model_Dense_%d.h5'%(k))

                # Prediction stage
                obj = Image_reconstruction(net, len(np.unique(labels)), patch_size = patch_size, overlap_percent = 0.10)
                predict_probs = obj.Inference(I)
                predict_labels = predict_probs.argmax(axis=-1)
                print('Predicted labels')
                print(np.unique(predict_labels, return_counts=True))
            
                test_mask_sample = np.zeros_like(mask)
                test_mask_sample[mask == 0] = 1             # Get test regions
                test_mask_sample[labels == 0] = 0           # Avoid non labeled pixels

                y_pred = predict_labels[test_mask_sample == 1]
                y_true = labels[test_mask_sample == 1]

                print('ypred:')
                print(np.unique(y_pred, return_counts=True))
                print('ytrue:')
                print(np.unique(y_true, return_counts=True))

                lt = 'a' 
                # if not k:
                #     lt = 'w'
                if Arq == 1:
                    file_metrics = open(fold_dir + "/metrics_Segnet.txt", lt)
                elif Arq == 2:
                    file_metrics = open(fold_dir + "/metrics_Unet.txt", lt)
                elif Arq == 3:
                    file_metrics = open(fold_dir + "/metrics_DeepLab.txt", lt)
                else:
                    file_metrics = open(fold_dir + "/metrics_FCDenseNet.txt", lt)

                O_Accuracy = 100 * accuracy_score(y_true, y_pred)
                print('O_Accuracy: ', np.around(O_Accuracy, decimals=1))
                if not k:
                    file_metrics.write('_ _ _  _ _ _ _\n\n')
                file_metrics.write('_ _ _ Run: %d___lr(%f)   batch_size(%d)   scale(%d)\n'%(k, lr, batch_size, scale))
                file_metrics.write('O_Accuracy: '); file_metrics.write(str(np.around(O_Accuracy, decimals=1)));  file_metrics.write('\n')

                # ____Metrics per class____
                
                F1_s = 100 * f1_score(y_true, y_pred, average=None)
                Precision = 100 * precision_score(y_true, y_pred, average=None)
                Recall = 100 * recall_score(y_true, y_pred, average=None)

                print('_____Metrics per class_____')
                print('Clases:     ', np.unique(y_pred))
                print('F1score:    ', np.around(F1_s, decimals=1))
                print('Precision:  ', np.around(Precision, decimals=1))
                print('Recall:     ', np.around(Recall, decimals=1))
                
                file_metrics.write('_____Metrics per class_____\n')
                file_metrics.write('Clases:     '); file_metrics.write(str(np.unique(y_true)));      file_metrics.write('\n')
                file_metrics.write('F1score:    '); file_metrics.write(str(np.around(F1_s, decimals=1)));        file_metrics.write('\n')
                file_metrics.write('Precision:  '); file_metrics.write(str(np.around(Precision, decimals=1)));   file_metrics.write('\n')
                file_metrics.write('Recall:     '); file_metrics.write(str(np.around(Recall, decimals=1)));      file_metrics.write('\n\n')

                # ____Average Metrics Positive classes only____

                if len(F1_s) == 4:
                    F1_s = F1_s[1:]
                    Precision = Precision[1:]
                    Recall = Recall[1:]
                
                F1_s_m = F1_s.mean()
                Precision_m = Precision.mean()
                Recall_m = Recall.mean()

                print('_____Average Metrics Positive classes only______')
                print('F1score:    ', np.around(F1_s_m, decimals=1))
                print('Precision:  ', np.around(Precision_m, decimals=1))
                print('Recall:     ', np.around(Recall_m, decimals=1))

                file_metrics.write('_____Average Metrics Positive classes only_____\n')
                file_metrics.write('F1score:    %.1f\n' %(np.around(F1_s_m, decimals=1)))
                file_metrics.write('Precision:  %.1f\n' %(np.around(Precision_m, decimals=1)))
                file_metrics.write('Recall:     %.1f\n\n\n' %(np.around(Recall_m, decimals=1)))

                file_metrics.close()

                del net
