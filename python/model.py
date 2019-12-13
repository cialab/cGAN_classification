from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import h5py
import sys
from PIL import Image
import random

from ops import *
from utils import *

class pix2pix(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=3, output_c_dim=3, dataset_name='facades',
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

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

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
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_B = self.real_data[:, :, :, :self.input_c_dim]
        self.real_A = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.real_A)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)
        self.real_A_sum = tf.summary.image("real_A", self.real_A) # ADDED THIS IN; HOPE IT WORKS
        self.real_B_sum = tf.summary.image("real_B", self.real_B) # Added THIS IN; HOPE IT WORKS

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def load_random_samples(self,data,ustd):
        #datamat = glob('{}/validation/*.h5'.format(self.dataset_name)) #self.batch_size
        #f=h5py.File(datamat[0])
        #arrays={}
        #for k,v in f.items():
        #    arrays[k]=np.array(v,dtype='float32')
        #data=arrays['patches']
        #data=np.swapaxes(data,0,3)
        #data=np.swapaxes(data,1,2)
        sample = load_data(data,self.batch_size,ustd=ustd)
        #f.close()

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, idx, data, ustd):
        sample_images = self.load_random_samples(data,ustd)
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        save_images(samples, [self.batch_size, 1],
                    '{}/train_{:05d}.png'.format(sample_dir, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum,
            self.fake_B_sum, self.real_A_sum, self.real_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("{}/logs".format(self.checkpoint_dir), self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        datamat = glob('{}/training/*.h5'.format(self.dataset_name))
        iterations = count(datamat[0]) #int(args.iterations) #min(count(datamat[0]), args.train_size) // self.batch_size

        f=h5py.File(datamat[0])
        arrays={}
        for k,v in f.items():
            arrays[k]=np.array(v,dtype='float32')
        data=arrays['patches']
        data=np.swapaxes(data,0,3)
        data=np.swapaxes(data,1,2)
        f.close()

        datamat = glob('{}/validation/*.h5'.format(self.dataset_name)) #self.batch_size
        f=h5py.File(datamat[0])
        arrays={}
        for k,v in f.items():
            arrays[k]=np.array(v,dtype='float32')
        datav=arrays['patches']
        datav=np.swapaxes(datav,0,3)
        datav=np.swapaxes(datav,1,2)
        f.close()

        # Get mean and std
        if args.ustd is None:
          ustd=None
        else:
          datamat=glob('{}/*.h5'.format(self.dataset_name))
          f=h5py.File(datamat[0])
          arrays={}
          for k,v in f.items():
            arrays[k]=np.array(v,dtype=np.float32)
          uA=np.swapaxes(arrays['uA'],0,2)
          stdA=np.swapaxes(arrays['stdA'],0,2)
          uB=np.swapaxes(arrays['uB'],0,2)
          stdB=np.swapaxes(arrays['stdB'],0,2)
          ustd=np.concatenate((uA,stdA,uB,stdB),axis=0)

        for e in xrange(0, args.epoch):
            for idx in xrange(0, iterations):
                batch = load_data(data, self.batch_size, ustd=ustd)
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images})

                counter += 1
                print("Epoch [%2d/%2d], Iteration [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (e, args.epoch, idx, iterations,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    self.sample_model(args.sample_dir, idx, datav, ustd)

                if np.mod(counter, 1000) == 2:
                    self.save(args.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
           # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        #model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        #model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # Read data
        datamat = glob('{}/'.format(self.dataset_name)+'{}/'.format(args.trvl)+'*.h5')
        f=h5py.File(datamat[0])
        arrays={}
        for k,v in f.items():
            arrays[k]=np.array(v,dtype='float32')
        data=arrays['patches']
        data=np.swapaxes(data,0,3)
        data=np.swapaxes(data,1,2)
        f.close()

        # How many to test
        testn=borm(data.shape[-1], args.test_size)
        sample_images = load_data(data, testn, seed=args.seed, is_test=True)

        # Shape
        sample_images = np.array(sample_images).astype(np.float32)

        # Add extra at end
        if args.batch_size!=1:
          addn = args.batch_size - testn%args.batch_size
          sample_images = np.concatenate((sample_images, np.zeros((addn,)+sample_images.shape[1:],dtype=sample_images.dtype)),axis=0)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Check if ustd
        if not args.ustd is None:
          savethis=np.zeros((testn,256,256,3),dtype=np.float32)
          for i, sample_image in enumerate(sample_images):
            print("sampling image ", i, testn)
            samples = self.sess.run(
              self.fake_B_sample,
              feed_dict={self.real_data: sample_image}
            )
            savethis[i]=np.squeeze(samples)
          f=h5py.File('{}/{}.h5'.format(args.test_dir, args.trvl))
          f.create_dataset('o',data=np.squeeze(sample_images[:,:,:,:,0:3]))
          f.create_dataset('c',data=np.squeeze(sample_images[:,:,:,:,3:6]))
          f.create_dataset('r',data=savethis)
          return

        # Inference time
        for i, sample_image in enumerate(sample_images):
            idx = i+1
            
            print("sampling image ", idx, testn)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            save_images(samples, [self.batch_size, 1],
                        '{}/{}_r_{:04d}.png'.format(args.test_dir, args.trvl, idx))
            save_images(sample_image[:,:,:,0:3], [self.batch_size, 1],
                        '{}/{}_o_{:04d}.png'.format(args.test_dir, args.trvl, idx))
            save_images(sample_image[:,:,:,3:6], [self.batch_size, 1],
                        '{}/{}_c_{:04d}.png'.format(args.test_dir, args.trvl, idx))

    def test2(self,args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # Testing train
        datamat2 = glob('{}/*.h5'.format(self.dataset_name))
        f2=h5py.File(datamat2[0])
        arrays2={}
        for k,v in f2.items():
            arrays2[k]=np.array(v,dtype='float32')
        data2=arrays2['patches']
        data2=np.swapaxes(data2,0,3)
        data2=np.swapaxes(data2,1,2)

        # sort testing input
        #n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
        #sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        testn2=borm(data2.shape[-1], args.test_size)
        sample_images2 = load_data(data2, testn2, sampling_seed=args.sampling_seed, discard_seed=args.discard_seed, is_test=True, discard=args.discard)

        if (self.is_grayscale):
            sample_images2 = np.array(sample_images2).astype(np.float32)[:, :, :, None]
        else:
            sample_images2 = np.array(sample_images2).astype(np.float32)

        # Add blanks at the end to make divisible by 100
        print(sample_images2.shape)
        nadd=args.batch_size-testn2%self.batch_size
        sample_images2=np.concatenate((sample_images2,np.zeros((nadd,)+sample_images2.shape[1:],dtype=sample_images2.dtype)),axis=0)
        print(sample_images2.shape)

        sample_images2 = [sample_images2[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images2), self.batch_size)]
        sample_images2 = np.array(sample_images2)
        print(sample_images2.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, sample_image in enumerate(sample_images2):
            idx = i+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            save_images(samples, [self.batch_size, 1],
                        '{}/val_r_{:04d}.png'.format(args.test_dir, idx))
            save_images(sample_image[:,:,:,0:3], [self.batch_size, 1],
                        '{}/val_o_{:04d}.png'.format(args.test_dir, idx))
            save_images(sample_image[:,:,:,3:6], [self.batch_size, 1],
                        '{}/val_c_{:04d}.png'.format(args.test_dir, idx))

    def test_fix(self,args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # Original training set
        datamat = glob('{}/training/*.h5'.format(args.ot_name))
        f=h5py.File(datamat[0])
        arrays={}
        for k,v in f.items():
            arrays[k]=np.array(v,dtype='uint8')
        data=arrays['patches']
        data=np.swapaxes(data,0,3)
        data=np.swapaxes(data,1,2)
        f.close()

        # Training set
        datamat2 = glob('{}/training/*.h5'.format(self.dataset_name))
        f2=h5py.File(datamat2[0])
        arrays2={}
        for k,v in f2.items():
            arrays2[k]=np.array(v,dtype='uint8')
        data2=arrays2['patches']
        data2=np.swapaxes(data2,0,3)
        data2=np.swapaxes(data2,1,2)
        f2.close()

        # Validation set
        datamat3 = glob('{}/validation/*.h5'.format(self.dataset_name))
        f3=h5py.File(datamat3[0])
        arrays3={}
        for k,v in f3.items():
            arrays3[k]=np.array(v,dtype='uint8')
        data3=arrays3['patches']
        data3=np.swapaxes(data3,0,3)
        data3=np.swapaxes(data3,1,2)
        f3.close()

        print('Finished reading datamats')

        # Load model
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Loop through each in training set
        o=np.zeros((data2.shape),dtype=np.uint8)
        r=np.zeros((data2.shape),dtype=np.uint8)
        for i in range(data2.shape[-1]):
          # Randomizer for picking origianl training set samples
          s=np.arange(data.shape[-1])
          np.random.shuffle(s)

          # Sample origianl training set and stick to current
          b=np.concatenate((data[:,:,:,s[0:args.batch_size-1]],np.expand_dims(data2[:,:,:,i],axis=3)),axis=3)
          b=np.array(b,dtype=np.float32)

          print('Training ',i,data2.shape[-1])
          abatch=np.zeros((args.batch_size,self.image_size,self.image_size,6),dtype=np.float32)
          for ii in range(args.batch_size):
            imgA=b[:,:,:,ii]

            # Condition
            nuke=np.random.rand(self.image_size,self.image_size).astype(np.float32)>float(args.discard)/100.0
            nuke=np.repeat(nuke[:,:,np.newaxis],3,axis=2)
            imgB=np.multiply(imgA,nuke)

            # Map values
            imgA=imgA/127.5 - 1.
            imgB=imgB/127.5 - 1.

            # Concatenate and assign to input
            imgAB=np.concatenate((imgA, imgB), axis=2)
            abatch[ii,:,:,:]=imgAB

          # Shape into a batch
          abatch=[abatch[ii:ii+args.batch_size]
            for ii in xrange(0,abatch.shape[0], args.batch_size)]
          abatch=np.array(abatch,dtype=np.float32)

          # Feed a batch
          for ix, sample_image in enumerate(abatch):
            idx = ix+1
            samples = self.sess.run(
              self.fake_B_sample,
              feed_dict={self.real_data: sample_image}
            )

            # Remap to between 0 and 1 and squeeze
            ri=samples[args.batch_size-1,:,:,:]
            ri=np.divide(np.add(ri,1.),2.)
            ri=np.squeeze(ri)
            ri=np.array(ri*255,dtype=np.uint8)

            # Put into the collection
            o[:,:,:,i]=data2[:,:,:,i]
            r[:,:,:,i]=ri

            # For testing purposes
            #scipy.misc.imsave('{}/train_o_{:04d}.png'.format(args.test_dir,i),o[:,:,:,i])
            #scipy.misc.imsave('{}/train_r_{:04d}.png'.format(args.test_dir,i),r[:,:,:,i])

        # Save images
        for i in range(o.shape[-1]):
          scipy.misc.imsave('{}/train_o_{:05d}.png'.format(args.test_dir,i),o[:,:,:,i])
          scipy.misc.imsave('{}/train_r_{:05d}.png'.format(args.test_dir,i),r[:,:,:,i])

        # Loop through each in validation set
        o=np.zeros((data3.shape),dtype=np.uint8)
        r=np.zeros((data3.shape),dtype=np.uint8)
        for i in range(data3.shape[-1]):
          # Randomizer for picking origianl training set samples
          s=np.arange(data.shape[-1])
          np.random.shuffle(s)

          # Sample origianl training set and stick to current
          b=np.concatenate((data[:,:,:,s[0:args.batch_size-1]],np.expand_dims(data3[:,:,:,i],axis=3)),axis=3)
          b=np.array(b,dtype=np.float32)

          print('Validation ',i,data3.shape[-1])
          abatch=np.zeros((args.batch_size,self.image_size,self.image_size,6),dtype=np.float32)
          for ii in range(args.batch_size):
            imgA=b[:,:,:,ii]

            # Condition
            nuke=np.random.rand(self.image_size,self.image_size).astype(np.float32)>float(args.discard)/100.0
            nuke=np.repeat(nuke[:,:,np.newaxis],3,axis=2)
            imgB=np.multiply(imgA,nuke)

            # Map values
            imgA=imgA/127.5 - 1.
            imgB=imgB/127.5 - 1.

            # Concatenate and assign to input
            imgAB=np.concatenate((imgA, imgB), axis=2)
            abatch[ii,:,:,:]=imgAB

          # Shape into a batch
          abatch=[abatch[ii:ii+args.batch_size]
            for ii in xrange(0,abatch.shape[0], args.batch_size)]
          abatch=np.array(abatch,dtype=np.float32)

          # Feed a batch
          for ix, sample_image in enumerate(abatch):
            idx = ix+1
            samples = self.sess.run(
              self.fake_B_sample,
              feed_dict={self.real_data: sample_image}
            )

            # Remap to between 0 and 1 and squeeze
            ri=samples[args.batch_size-1,:,:,:]
            ri=np.divide(np.add(ri,1.),2.)
            ri=np.squeeze(ri)
            ri=np.array(ri*255,dtype=np.uint8)

            # Put into the collection
            o[:,:,:,i]=data3[:,:,:,i]
            r[:,:,:,i]=ri

            # For testing purposes
            #scipy.misc.imsave('{}/val_o_{:04d}.png'.format(args.test_dir,i),o[:,:,:,i])
            #scipy.misc.imsave('{}/val_r_{:04d}.png'.format(args.test_dir,i),r[:,:,:,i])

        # Save images
        for i in range(o.shape[-1]):
          scipy.misc.imsave('{}/val_o_{:05d}.png'.format(args.test_dir,i),o[:,:,:,i])
          scipy.misc.imsave('{}/val_r_{:05d}.png'.format(args.test_dir,i),r[:,:,:,i])

    def segment(self,args):

      # Get the HPF being tested as well as the rest from the validation set
      l=args.hpfs.split(",")
      hpf_0=l[0]
      hpf_=l[1:]

      # Read in image for segmentation and create masks/accumulators 
      im=Image.open(args.hpf_dir+'/'+hpf_0)
      width=im.size[0]
      height=im.size[1]
      tumor_mask=np.zeros((height,width),dtype=np.float32)
      nontumor_mask=np.zeros((height,width),dtype=np.float32)
      tumor_count=np.zeros((height,width),dtype=np.float32)
      nontumor_count=np.zeros((height,width),dtype=np.float32)
      tumor_r=np.zeros((height,width),dtype=np.uint8)
      nontumor_r=np.zeros((height,width),dtype=np.uint8)

      # Randomize our query for hpf_0
      xs=range(0,width-self.image_size+8,8)
      ys=range(0,height-self.image_size+8,8)
      xv,yv=np.meshgrid(xs,ys)
      xys=[[0,0]]*len(xs)*len(ys)
      i=0
      for x in range(0,len(xs)):
        for y in range(0,len(ys)):
          xys[i]=[xv[y,x],yv[y,x]]
          i=i+1

      # Create containers for hpf_0 tile reconstructions
      r1=np.zeros((len(xys),self.image_size,self.image_size,3),dtype=np.float32)
      r2=np.zeros((len(xys),self.image_size,self.image_size,3),dtype=np.float32)

      # Loop through hpf_0 indexes
      idxs=range(0,len(xys),args.hpf_sample)
      idxs.append(len(xys))
      # Loop around each args.hpf_sample
      for i in range(0,len(idxs)-1):
        hpf_0_tiles=np.zeros((idxs[i+1]-idxs[i],self.image_size,self.image_size,3),dtype=np.float32)
        # Get args.hpf_sample
        for ii in range(idxs[i],idxs[i+1]):
          hpf_0_tiles[ii,:,:,:]=im.crop((xys[ii][0],xys[ii][1],xys[ii][0]+self.image_size,xys[ii][1]+self.image_size))

        # Loop around the rest of the images
        hpf__tiles=np.zeros((len(hpf_)*args.hpf_sample,self.image_size,self.image_size,3),dtype=np.float32)
        for ii in range(0,len(hpf_)):
          im2=Image.open(args.hpf_dir+'/'+hpf_[i])
          xys2=random.sample(xys,args.hpf_sample)
          # Sample image
          for iii in range(0,len(xys2)):
            hpf__tiles[ii*args.hpf_sample+iii,:,:,:]=im2.crop((xys2[iii][0],xys2[iii][1],xys2[iii][0]+self.image_size,xys2[iii][1]+self.image_size))
        # At this point, we have sampled all the 'other' images
        # Now, we concatenate the tiles from hpf_0 and the tiles from the 'other' HPFs
        tiles=np.concatenate((hpf_0_tiles,hpf__tiles),axis=0)

        # Fuck what do I do now???
        # I think I'm going to make a random array for batching purposes
        # Fuuuuuuuuuck

        # Add blanks at end (maybe mean patch in the future?)
        nadd=self.batch_size-len(tiles)%self.batch_size
        tiles=np.concatenate((tiles,np.zeros((nadd,)+tiles.shape[1:],dtype=tiles.dtype)),axis=0)

        # Random array for indexing of batches
        tiles_idx=random.sample(range(0,tiles.shape[0]),tiles.shape[0])

        # Shape into batches randomly
        tiles=[tiles[tiles_idx[i:self.batch_size]]
                         for i in xrange(0, len(tiles), self.batch_size)]
        tiles=np.array(tiles)

        # Load checkpoint
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Being feeding batches
        tiles_r1=np.zeros(tiles.shape,dtype=np.float32)
        for ix, sample_image in enumerate(tiles):
            idx = ix+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            tiles_r1[ix*self.batch_size:(ix+1)*self.batch_size,:,:,:]=samples

        # Holy christ I hope this is right
        r1[idxs[i+1]-idxs[i],:,:,:]=tiles_r1[tiles_idx[0:hpf_0_tiles.shape[0]],:,:,:]

        # Load checkpoint
        if self.load(args.checkpoint_dir2):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Being feeding batches
        tiles_r2=np.zeros(tiles.shape,dtype=np.float32)
        for ix, sample_image in enumerate(tiles):
            idx = ix+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            tiles_r2[ix*self.batch_size:(ix+1)*self.batch_size,:,:,:]=samples

        # Holy christ I hope this is right
        r2[idxs[i+1]-idxs[i],:,:,:]=samples[tiles_idx[0:hpf_0_tiles.shape[0]],:,:,:]

      # Now to compare the reconstructions; hopefully hpf_0_tiles, r1, and r1 have the same indexes
      for ii in range(0,hpf_0_tiles.shape[0]):
        d1=np.mean((hpf_0_tiles[ii,:,:,0:3]-r1[ii,:,:,:])**2)
        d2=np.mean((hpf_0_tiles[ii,:,:,0:3]-r2[ii,:,:,:])**2)
        p=abs(d1-d2)/(d1+d2)
        if d1>d2:
          tumor_mask[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]=tumor_mask[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]+p
          tumor_count[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]=tumor_count[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]+1
          nontumor_mask[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]=nontumor_mask[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]+(1-p)
          nontumor_count[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]=nontumor_count[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]+1
          #hpf_0_r[[xys[i][0],xys[i][1]]=r1[i,:,:,:]
        else:
          nontumor_mask[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]=nontumor_mask[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]+p
          nontumor_count[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]=nontumor_count[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]+1
          tumor_mask[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]=tumor_mask[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]+(1-p)
          tumor_count[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]=tumor_count[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]+1
          #hpf_0_r[[xys[i][0],xys[i][1]]=r2[i,:,:,:] 

        # Reconstruction by both models
        tumor_r[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]=r1[ii,:,:,:]
        nontumor_r[xys[ii][0]:xys[ii][0]+self.image_size,xys[i][1]:xys[ii][1]+self.image_size]=r2[ii,:,:,:]

      # Home stretch?
      tumor_mask=Image.fromarray((np.divide(tumor_mask,tumor_count).astype(np.uint8)))
      nontumor_mask=Image.fromarray((np.divide(nontumor_mask,nontumor_count).astype(np.uint8)))
      tumor_r=Image.fromarray((tumor_r))
      nontumor_r=Image.fromarray((nontumor_r))

      # Save
      tumor_mask.save(args.hpf_dir+'/'+hpf_0+'_tumor.png')
      nontumor_mask.save(args.hpf_dir+'/'+hpf_0+'_nontumor.png')
      tumor_r.save(args.hpf_dir+'/'+hpf_0+'_tumor_r.png')
      nontumor_r.save(args.hpf_dir+'/'+hpf_0+'_nontumor_r.png')
