import argparse
import os
import scipy.misc
import numpy as np

from model import pix2pix
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
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
parser.add_argument('--discard_seed', dest='discard_seed', type=int, default=-1, help='seed desired for pulling testing samples: -1 for no seed')
parser.add_argument('--seed', dest='seed', type=int, default=-1, help='seed desired for what to discard')
parser.add_argument('--discard', dest='discard', type=int, default=60, help='how much data to discard from patches')
parser.add_argument('--test_size', dest='test_size', type=int, default=100, help='how many images from each set (tr,vl,te) you want to test')
parser.add_argument('--test_slide', dest='slide_test', type=bool, default=False, help='do you want to test only one slide?')
parser.add_argument('--big_test', dest='big_test', type=bool, default=False, help='is this really fucking annoying or what?')
parser.add_argument('--test_hpf', dest='test_hpf', type=bool, default=False, help='segment an hpf')
parser.add_argument('--hpf_dir', dest='hpf_dir', default='/fs/scratch/osu8705/tumor_budding_HPFs/', help='directory of HPFs')
parser.add_argument('--hpfs', dest='hpfs', default='10A\ H\&E_1.tif', help='list of hpfs, separated by commas, first one is the one segmented')
parser.add_argument('--hpf_sample', dest='hpf_sample', type=int, default=1000, help='how many tiles sampled from each HPF during segmentation')
parser.add_argument('--checkpoint_dir2', dest='checkpoint_dir2', default='./checkpoint', help='other class model saved here')
parser.add_argument('--ot_name', dest='ot_name', default='facades', help='original training set location')
parser.add_argument('--trvl', dest='trvl', default='validation', help='test training or test validation')
parser.add_argument('--ustd', dest='ustd', default=None, help='subtract that mean and std?')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = pix2pix(sess, image_size=args.fine_size, batch_size=args.batch_size,
                        output_size=args.fine_size, dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)

        if args.phase == 'train':
            model.train(args)
        elif args.slide_test:
            model.test2(args)
        elif args.big_test:
            model.test_fix(args)
        elif args.test_hpf:
            model.segment(args)
        else:
            model.test(args)

if __name__ == '__main__':
    tf.app.run()
