import argparse
import os
import scipy.misc
import numpy as np
import sys

from model import pix2pix

import tensorflow as tf
parser = argparse.ArgumentParser(description='')

parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=20000, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=143, help='scale images to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=2, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=7, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test, generate_image, create_dataset')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')

#####___No@___#####

parser.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
parser.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
# Argument: --number_of_classes = --output_nc for using Deeplabv3
parser.add_argument("--number_of_classes", type=int, default=7, help="Number of classes to be predicted.")
parser.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
parser.add_argument("--resnet_model", default="resnet_v2_50", choices=["resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "resnet_v2_200"], help="Resnet model to use as feature extractor. Choose one of: resnet_v2_50 or resnet_v2_101")

parser.add_argument("--sampling_type", default="dropout", choices=["dropout", 'none'], help="Noise used in the GAN")
parser.add_argument("--generator", default="deeplab", choices=["deeplab", "unet"], help="Generator's architecture ")
parser.add_argument("--discriminator", default="atrous", choices=["atrous", "pix2pix"], help="Discriminator's architecture ")
parser.add_argument("--data_augmentation", type=bool, default=True, choices=[True, False], help="Data Augmentation Flag")

parser.add_argument('--dataset_name', dest='dataset_name', default='LEM', choices=["LEM", "Campo_Verde"], help='name of the dataset')

parser.add_argument("--norm_type", default="min_max", choices=["min_max", "std", "wise_frame_mean"], help="Type of normalization ")
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument("--output_stride", type=int, default=16, help="Spatial Pyramid Pooling rates")
parser.add_argument("--patch_overlap", type=float, default=0.92, help="Overlap percentage between patches")

#####_________#####

args = parser.parse_args()

def actions():

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        model = pix2pix(sess, args, image_size=args.fine_size, load_size=args.load_size, batch_size=args.batch_size,
                        output_size=args.fine_size, input_c_dim=args.input_nc,
                        output_c_dim=args.output_nc, dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir) #args added for using DeepLabv3

        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'generate_image':
            model.generate_image(args)
        else:
            print ('...')

def main(_):

    actions()


if __name__ == '__main__':
    tf.app.run()
