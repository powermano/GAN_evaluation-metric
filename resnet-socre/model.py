# --------------------------------------------------------
# Tensorflow Resnet
# Licensed under The MIT License [see LICENSE for details]
# Written by Tao Cai
# --------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import numpy as np
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import regularizers, \
    initializers, layers
import cv2
import os
import os.path
import sys
import tarfile
from six.moves import urllib
import glob
import scipy.misc
import math
from PIL import Image

def resnet_arg_scope(
        is_training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=initializers.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


is_training = True
global_pool = True
spatial_squeeze = True
num_classes = 1000
image = tf.placeholder(tf.float32, shape=[None, None, None, 3])
depth_multiplier=1
min_base_depth=8
check_dir = r'D:\用户目录\下载\resnet_v1_50_2016_08_28'



depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
blocks = [
      resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
                      stride=2),
      resnet_v1_block('block2', base_depth=depth_func(128), num_units=4,
                      stride=2),
      resnet_v1_block('block3', base_depth=depth_func(256), num_units=6,
                      stride=2),
      resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
                      stride=1),
       ]
print('1')
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    with tf.variable_scope('resnet_v1_50', 'resnet_v1_50'):
        net = resnet_utils.conv2d_same(
            image, 64, 7, stride=2, scope='conv1')
        net = slim.max_pool2d(
            net, [3, 3], stride=2, padding='SAME', scope='pool1')
    net, _ = resnet_v1.resnet_v1(
        net, blocks[0:1], global_pool=False, include_root_block=False,
        scope='resnet_v1_50')

with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
    net_conv3, _ = resnet_v1.resnet_v1(
        net, blocks[1:2], global_pool=False, include_root_block=False,
        scope='resnet_v1_50')

with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
    net_conv4, _ = resnet_v1.resnet_v1(
        net_conv3, blocks[2:3], global_pool=False,
        include_root_block=False, scope='resnet_v1_50')

with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
    net_conv5, _ = resnet_v1.resnet_v1(
        net_conv4, blocks[-1:], global_pool=False,
        include_root_block=False, scope='resnet_v1_50')

initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

with tf.variable_scope('resnet_v1_50', 'resnet_v1_50'):
    if global_pool:
        # Global average pooling.
        net_global = tf.reduce_mean(net_conv5, [1, 2], name='pool5', keep_dims=True)



    if num_classes:
        net_final = slim.conv2d(net_global, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')
        if spatial_squeeze:
            net_squzee = tf.squeeze(net_final, [1, 2], name='SpatialSqueeze')
        cls_pred = tf.arg_max(net_squzee, 1)
        predicted = slim.softmax(net_squzee, scope='predictions')


saver = tf.train.Saver()
init = tf.global_variables_initializer()


def get_image_list(image_list_path):
    image_id = os.listdir(image_list_path)
    img_list = []
    for id in image_id:
        img = Image.open(os.path.join(image_list_path, id))
        img = img.resize((224, 224))
        im_array = np.array(img, dtype=np.float32)
        # im_array = np.expand_dims(im_array, axis=0)
        img_list.append(im_array)
    print('Suessfully get image list')

    return img_list

def test_per_image(image_path):

    img = Image.open(image_path)
    img = img.resize((224,224))
    #img.show()
    print(img.size)
    im_array = np.array(img, dtype=np.float32)
    im_array = np.expand_dims(im_array, axis=0)
    return im_array

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
    assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    print('Starting calculating inception scores')
    inps = []
    for img in images:
      img = img.astype(np.float32)
      inps.append(np.expand_dims(img, 0))
    bs = 1
    # with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
      sys.stdout.write(".")
      sys.stdout.flush()
      inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
      inp = np.concatenate(inp, 0)
      pred = sess.run(predicted, feed_dict={image: inp})
      preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    print('Sucessfully get inception scores')
    return np.mean(scores), np.std(scores)

with tf.Session() as sess:
    sess.run(init)
    #chkpt_fname = tf.train.latest_checkpoint(check_dir)
    chkpt_fname = r'E:\resnet_v1_50.ckpt'
    print(chkpt_fname)
    print('loading model ...............')
    saver.restore(sess, chkpt_fname)
    print('Suessfully load the model................')
    # img = test_per_image(r'E:/cat.jpg')
    # cls, output, sf_out = sess.run([cls_pred, net_final, predicted], feed_dict={image: img})
    # print(cls)
    # print(output)
    # print(sf_out)
    path_dir = r'E:\input\wreck\trainB'
    image_list = get_image_list(path_dir)
    mean_score, mean_std = get_inception_score(image_list)




