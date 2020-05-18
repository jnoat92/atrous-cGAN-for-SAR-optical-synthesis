from __future__ import division
import os
import time
import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from sklearn import preprocessing as pre
from sklearn.externals import joblib
import scipy.io as io
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
import scipy.io as sio

from ops import *
from utils import *
#from saveweigths import *

#####___No@___#####
import network
import sys
slim = tf.contrib.slim
#####_________#####

class pix2pix(object):
    def __init__(self, sess, args, image_size=256, load_size=286,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=11, output_c_dim=7, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None):

        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.load_size = load_size
        self.fine_size = image_size

        self.sar_path = '../datasets/' + args.dataset_name + '/Sentinel_1/'
        self.opt_path = '../datasets/' + args.dataset_name + '/Landsat_8/Images/'
        self.lbl_path = '../datasets/' + args.dataset_name + '/Landsat_8/Reference/'
        self.cloud_path = '../datasets/' + args.dataset_name + '/Landsat_8/Cloud_mask/'

        if args.dataset_name == 'Amazonia_Legal':
            self.sar_name_t0 = '2018_07'
            self.opt_name_t0 = '24_07_2018_image'
            self.lbl_name_t0 = 'REFERENCE_2018_EPSG32620'

            self.sar_name_t1 = '2019_07'
            self.opt_name_t1 = '27_07_2019_image'
            self.lbl_name_t1 = 'REFERENCE_2019_EPSG32620'

        elif args.dataset_name == 'Cerrado_biome':
            self.sar_name_t0 = '2017_08'
            self.opt_name_t0 = '18_08_2017_image'
            self.lbl_name_t0 = 'REFERENCE_2018_EPSG4674' #change

            self.sar_name_t1 = '2018_08'
            self.opt_name_t1 = '21_08_2018_image'
            self.lbl_name_t1 = 'REFERENCE_2018_EPSG4674'

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda
        
        self.args = args
        self.sampling_type = args.sampling_type
        self.norm_type = args.norm_type
        self.keep_prob = 0.5

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bnr = batch_norm(name='d_bnr')
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')

        self.g_bn_er = batch_norm(name='g_bn_er')
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()


    def build_model(self):
        self.bands_sar = 2

        # Picking up the generator and discriminator
        generator = getattr(network, self.args.generator)
        discriminator = getattr(network, self.args.discriminator + '_discriminator')

        self.sar_t0 = tf.placeholder(tf.float32,
                                    [None, 3*self.image_size, 3*self.image_size, self.bands_sar],
                                    name='sar_t0')
        self.opt_t0 = tf.placeholder(tf.float32,
                                    [None, self.image_size, self.image_size, self.output_c_dim],
                                    name='opt_t0')

        self.SAR = self.sar_t0
        self.OPT = self.opt_t0
        self.fake_opt_t0 = generator(self, self.SAR, None, reuse=False)
        self.fake_opt_t0_sample = generator(self, self.SAR, None, reuse=True)
        self.OPT_fake = self.fake_opt_t0
        
        self.D, self.D_logits = discriminator(self, self.SAR, self.OPT, reuse=False)
        self.D_, self.D_logits_ = discriminator(self, self.SAR, self.OPT_fake, reuse=True)
        
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.opt_t0 - self.fake_opt_t0))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        
        with tf.variable_scope('learning_rate'):
            self.Lr = tf.Variable(self.args.lr, trainable=False)
        self.d_optim = tf.train.AdamOptimizer(self.Lr, beta1=self.args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.Lr, beta1=self.args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        self.model = "%s_bs%s_%s_ps%s_op%d" % \
                      (self.args.discriminator, self.batch_size, self.norm_type, self.fine_size, self.args.patch_overlap*100)
        if self.args.mask == 'k-fold':
            self.model += '_fold_' + str(self.args.k)

        self.saver = tf.train.Saver()

        print('_____Generator_____')
        self.count_params(self.g_vars)
        print('_____Discriminator_____')
        self.count_params(self.d_vars)
        print('_____Full Model_____')
        self.count_params(t_vars)


    def train(self, args):
        """Train cGAN"""

        # Create dataset
        if args.date == 'both':
            train_patches, val_patches, \
            test_patches, self.data_dic = create_dataset_both_images(self)
        elif args.date == 'd0':
            self.sar_name = self.sar_name_t0
            self.opt_name = self.opt_name_t0
            self.lbl_name = self.lbl_name_t0
            train_patches, val_patches, \
                test_patches, self.data_dic = create_dataset_coordinates(self, prefix = 0)
        elif args.date == 'd1':
            self.sar_name = self.sar_name_t1
            self.opt_name = self.opt_name_t1
            self.lbl_name = self.lbl_name_t1
            train_patches, val_patches, \
                test_patches, self.data_dic = create_dataset_coordinates(self, prefix = 1)

        # Model
        model_dir = os.path.join(self.checkpoint_dir, args.dataset_name, self.model)
        sample_dir = os.path.join(model_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        # Initialize graph
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(model_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        idx = 601
        save_samples_multiresolution(self, test_patches, output_path=sample_dir, 
                                     idx=idx, epoch=0, labels=True)
        
        loss_trace_G, loss_trace_D = [], []
        errD_fake, errD_real, errG = 0, 0, 0
        aux_weight = np.zeros((self.fine_size, self.fine_size))

        counter = 0
        start_time = time.time()
        for epoch in xrange(counter, args.epoch):

            #e_dec = 30
            #if epoch > e_dec:
                #new_lr = args.lr - \
                         #args.lr*(epoch - e_dec)/(args.epoch - e_dec)
                #self.sess.run(tf.assign(self.Lr, new_lr))

            data = train_patches
            np.random.shuffle(data)
            batches = min(len(data), args.train_size) // self.batch_size

            for batch in xrange(0, batches):

                # Taking the Batch
                sar_t0, opt_t0 = [], []
                for im in xrange(batch*self.batch_size, (batch+1)*self.batch_size):
                    batch_image = Take_patches(data, idx=im,
                                               data_dic=self.data_dic,
                                               fine_size=self.fine_size,
                                               random_crop_transformation=True,
                                               labels=False)
                    sar_t0.append(batch_image[0])
                    opt_t0.append(batch_image[1])

                sar_t0 = np.asarray(sar_t0)
                opt_t0 = np.asarray(opt_t0)
                
                # Update D network
                _, summary_str = self.sess.run([self.d_optim, self.d_sum],
                                               feed_dict={self.sar_t0: sar_t0, self.opt_t0: opt_t0})

                # Update G network
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                for _ in range(2):
                    _ = self.sess.run([self.g_optim],
                                      feed_dict={self.sar_t0: sar_t0, self.opt_t0: opt_t0})
                
                if np.mod(batch + 1, 500) == 0:
                    errD_fake = self.d_loss_fake.eval({ self.sar_t0: sar_t0, self.opt_t0: opt_t0 })
                    errD_real = self.d_loss_real.eval({ self.sar_t0: sar_t0, self.opt_t0: opt_t0 })
                    errG = self.g_loss.eval({ self.sar_t0: sar_t0, self.opt_t0: opt_t0 })
                    print("Epoch: [%2d] [%4d/%4d] lr: %.6f time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, batch, batches, self.Lr.eval(),
                            time.time() - start_time, errD_fake+errD_real, errG))

            loss_trace_G.append(errG)
            loss_trace_D.append(errD_fake+errD_real)
            np.save(model_dir + '/loss_trace_G', loss_trace_G)
            np.save(model_dir + '/loss_trace_D', loss_trace_D)

            self.save(args.checkpoint_dir, epoch)
            # save sample
            save_samples_multiresolution(self, test_patches, output_path=sample_dir, 
                                        idx=idx, epoch=epoch)


    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"

        checkpoint_dir = os.path.join(checkpoint_dir, self.args.dataset_name, self.model)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        print("Saving checkpoint!")


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        print(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            aux = 'model_example'
            for i in range(len(ckpt_name)):
                if ckpt_name[-i-1] == '-':
                    aux = ckpt_name[-i:]
                    break
            return aux
        else:
            return ''


    def generate_image(self, args):

        if args.date == 'd0':
            self.sar_name = self.sar_name_t0
            self.opt_name = self.opt_name_t0
            self.lbl_name = self.lbl_name_t0
        elif args.date == 'd1':
            self.sar_name = self.sar_name_t1
            self.opt_name = self.opt_name_t1
            self.lbl_name = self.lbl_name_t1
        else:
            print('Specify Date !!')
            return

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        model_dir = os.path.join(self.checkpoint_dir, args.dataset_name, self.model)
        output_path = os.path.join(model_dir, 'samples')

        mod = self.load(model_dir)
        if mod:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        print( 'Generating Image for_' + args.dataset_name)

        # Percent of overlap between consecutive patches.
        # The overlap will be multiple of 2 and 3, this guarantes to
        # use the same variables to construct the optical image.
        P = 0.50
        overlap = 3 * round(self.image_size * P)
        overlap -= overlap % 6
        stride = 3 * self.image_size - overlap
        stride_opt   = stride // 3
        overlap_opt  = overlap // 3

        # Opening SAR Image
        print('sar_name: %s' %(self.sar_name))
        sar_path_t0 = self.sar_path + self.sar_name + '/' + self.sar_name + '.npy'
        sar_t0 = np.load(sar_path_t0).astype('float32')
        # Normalization
        scaler = joblib.load('../datasets/' + args.dataset_name + '/Norm_params_' + self.sar_name + \
                             '/sar_' + self.sar_name   + '_' + self.norm_type + '.pkl')
        num_rows, num_cols, _ = sar_t0.shape
        sar_t0 = sar_t0.reshape((num_rows * num_cols, -1))
        sar_t0 = scaler.transform(sar_t0)
        sar_t0 = sar_t0.reshape((num_rows, num_cols, -1))

        # Add Padding to the image to match with the patch size and the overlap
        step_row = (stride - num_rows % stride) % stride
        step_col = (stride - num_cols % stride) % stride

        pad_tuple = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)), (0,0) )
        sar_pad_t0 = np.pad(sar_t0, pad_tuple, mode = 'symmetric')

        # Number of patches: k1xk2
        k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
        print('Number of patches: %d x %d' %(k1, k2))

        # Taking the test mask
        mask = Image.open('../datasets/' + args.dataset_name + '/mask_train_val_test.tif')
        mask = np.array(mask)
        mask_test = mask == 0

        # Inference
        fake_pad_opt_t0 = np.zeros((k1*stride_opt, k2*stride_opt, self.output_c_dim))
        # for k in range(1):
        sample_name = self.model + "_" +  self.sar_name  # + "_" +  str(k)
        print(sample_name)
        test_patches_numb = 0
        start = time.time()
        for i in range(k1):
            for j in range(k2):

                mask_test_patch = mask_test[i*stride+overlap//2:(i*stride+overlap//2 + stride),
                                            j*stride+overlap//2:(j*stride+overlap//2 + stride)]
                if mask_test_patch.any():
                    test_patches_numb += 1
                
                sar_t0 = sar_pad_t0[i*stride:(i*stride + 3*self.image_size),
                                    j*stride:(j*stride + 3*self.image_size), :]
                sar_t0 = sar_t0.reshape(1, 3*self.image_size, 3*self.image_size, -1)

                fake_patch = self.sess.run(self.fake_opt_t0_sample,
                                           feed_dict={self.sar_t0: sar_t0})

                fake_pad_opt_t0[i*stride_opt : i*stride_opt+stride_opt, 
                                j*stride_opt : j*stride_opt+stride_opt, :] = fake_patch[0, overlap_opt//2 : overlap_opt//2 + stride_opt, 
                                                                                            overlap_opt//2 : overlap_opt//2 + stride_opt, :]
            
            print('row %d: %.2f min' %(i+1, (time.time() - start)/60))
        
        print('Number of patches that incorporate pixels to the test region: %d' %(test_patches_numb))

        # Taken off the padding
        rows = k1*stride_opt-step_row//3
        cols = k2*stride_opt-step_col//3
        fake_opt_t0 = fake_pad_opt_t0[:rows, :cols]
        
        print('fake_opt_t0 size: ');  print(fake_opt_t0.shape)
        print('Inference time: %.2f min' %((time.time() - start)/60))

        # Denomarlize
        if self.norm_type == 'wise_frame_mean':
            scaler = joblib.load('../datasets/' + args.dataset_name + '/Norm_params_' + self.sar_name + \
                                 '/opt_' + self.opt_name + '_' + 'std' + '.pkl')
        else:
            scaler = joblib.load('../datasets/' + args.dataset_name + '/Norm_params_' + self.sar_name + \
                                 '/opt_' + self.opt_name + '_' + self.norm_type + '.pkl')
        fake_opt_t0 = Denormalization(fake_opt_t0, scaler)

        # np.save(output_path + '/' + sample_name, fake_opt_t0)
        np.savez_compressed(output_path + '/' + sample_name, fake_opt_t0)
        sio.savemat(output_path + '/' + sample_name,  {sample_name: fake_opt_t0})


    def count_params(self, t_vars):
        """
        print number of trainable variables
        """
        n = np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])
        print("Model size: %dK params" %(n/1000))

        # w = self.sess.run(self.g_vars)
        # for val, var in zip(w, self.g_vars):
        #     if 'generator' in var.name:
        #         print(var.name)
        #         print(val.shape)
        # #         # break
        # sys.exit()
