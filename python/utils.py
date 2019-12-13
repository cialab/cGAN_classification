"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import h5py
import time

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def borm(maxx,test_size):
	return min(maxx,test_size)

def count(data):
    return int(data.split("/")[-1].split(".")[0])

def rand_datamat_and_idx(datamats):
    d=random.choice(datamats)
    n=d.split("/")[-1].split(".")[0]
    idx=random.randint(0,int(n))
    return d,idx

def load_data(data, batch_size, im_size=256, seed=-1, is_test=False, ustd=None):
    if is_test:
        if seed==0:
            samp=range(data.shape[-1])
        elif seed==-1:
            samp=random.sample(range(data.shape[-1]),batch_size)
        else:
            random.seed(seed)
            samp=random.sample(range(data.shape[-1]),batch_size)
    else:
        samp=random.sample(range(data.shape[-1]),batch_size)

    imgs=np.zeros((batch_size,im_size,im_size,2*3),dtype=np.float32)
    for i in range(batch_size):
        im=data[:,:,:,samp[i]]#load_image(datamats)
        img_A=im[:,:,0:3]
        img_B=im[:,:,3:]

        if ustd is None:
          img_A = img_A/127.5 - 1.   # originally 8-bit, 3 channel tif 
          img_B = img_B/127.5 - 1.  # originally binary
        else:
          img_A = img_A-ustd[0]
          img_A = np.divide(img_A,ustd[1])
          img_B = img_B-ustd[2]
          img_B = np.divide(img_B,ustd[3])

        img_AB = np.concatenate((img_A, img_B), axis=2)
        imgs[i]=img_AB
    return imgs

def load_alot_data(data, batch_size, i1, i2, im_size=256, sampling_seed=-1, is_test=False):
    #if is_test:
    #    if sampling_seed==-1:
    #        samp=random.sample(range(data.shape[-1]),data.shape[-1])
    #    else:
    #        random.seed(sampling_seed)
    #        samp=random.sample(range(data.shape[-1]),data.shape[-1])
    #else:
    #    samp=random.sample(range(data.shape[-1]),batch_size)

    imgs=np.zeros((batch_size,im_size,im_size,2*3),dtype=np.float32)
    for i in range(i2-i1):
        im=data[:,:,:,i]#load_image(datamats)
        img_A=scipy.misc.imresize(im[:,0:im.shape[0],:],[im_size,im_size],interp='bicubic')
        img_B=scipy.misc.imresize(im[:,im.shape[0]:,:],[im_size,im_size],interp='bicubic')
        img_A = img_A/127.5 - 1.
        img_B = (img_B-0.5)/0.5

        img_AB = np.concatenate((img_A, img_B), axis=2)
        imgs[i]=img_AB
    return imgs

def load_image(image_path,idx, fine_size=256):
    f=h5py.File(image_path)
    arrays={}
    for k,v in f.items():
        arrays[k]=np.array(v,dtype='float32')
    mat=arrays['patches']
    mat=np.swapaxes(mat,0,3)
    mat=np.swapaxes(mat,1,2)
    im=mat[:,:,:,idx]
    im=scipy.misc.imresize(im,[fine_size,fine_size],interp='bicubic')
    #input_img = imread(image_path)
    #w2 = int(w/2)
    #img_A = input_img[:, 0:w2]
    #img_B = input_img[:, w2:w]
    return im

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def save_image(image, size, image_path):
    print(image.shape)
    return scipy.misc.imsave(image_path, inverse_transform(image))

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


