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
# import cv2
from PIL import Image
import os

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
num_classes = 2
depth_multiplier = 1
min_base_depth = 8
batch_size = 1  # test_batch
img_height = 224
img_width = 224
img_layer = 3
img_size = img_height * img_width
image = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])

def model():
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

    return cls_pred, predicted


def test_per_image(image_path):
    cls, softmax_result = model()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    img = Image.open(image_path)
    img = img.resize((224, 224))
    # img.show()
    print(img.size)
    im_array = np.array(img, dtype=np.float32)
    im_array = np.expand_dims(im_array, axis=0)
    with tf.Session() as sess:
        sess.run(init)
        #chkpt_fname = tf.train.latest_checkpoint(check_dir)
        #chkpt_fname = r'E:\resnet_v1_50.ckpt'
        chkpt_fname = '/home/ct/resnet_v1_50.ckpt'
        print(chkpt_fname)
        print('loading model ...............')
        saver.restore(sess, chkpt_fname)
        print('Suessfully load the model................')
        # img = test_per_image(r'E:/dog.jpg')
        cls_pred, sf_out = sess.run([cls, softmax_result], feed_dict={image: im_array})
        print(cls_pred)
        print(sf_out.shape)
    # print(output)
    # print(sf_out)
    return softmax_result


def test_images(image_dir):
    cls, softmax_result = model()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        filenames_A = tf.train.match_filenames_once(os.path.join(image_dir, '*.jpg'))
        queue_length_A = tf.size(filenames_A)
        filename_queue_A = tf.train.string_input_producer(filenames_A)
        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        image_A = tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [224, 224])

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess.run(init)
        preds = []
        sf_out_pred = []
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        num_files_A = sess.run(queue_length_A)
        print('Totally %d images' % num_files_A)
        A_input = np.zeros((num_files_A, batch_size, img_height, img_width, img_layer))
        for i in range(num_files_A):
            image_tensor = sess.run(image_A)
            if (image_tensor.size == img_size * batch_size * img_layer):
                A_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layer))
        coord.request_stop()
        coord.join(threads)

        #chkpt_fname = tf.train.latest_checkpoint(check_dir)
        chkpt_fname = r'E:\resnet_v1_50.ckpt'
        #chkpt_fname = '/home/ct/resnet_v1_50.ckpt'
        print(chkpt_fname)
        print('loading model ...............')
        saver.restore(sess, chkpt_fname)
        print('Suessfully load the model................')
        # img = test_per_image(r'E:/dog.jpg')
        for i in range(num_files_A):
            cls_pred, sf_out = sess.run([cls, softmax_result], feed_dict={image: A_input[i]})
            preds.append(cls_pred)
            sf_out_pred.append(sf_out)
        return preds, sf_out_pred

def get_inception_socres(predictions, splits=10):
    print('Starting calculating inception scores')
    preds = np.concatenate(predictions, 0)
    print(preds.shape)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    print('Sucessfully get inception scores')
    return np.mean(scores), np.std(scores)


def train(image_dir):
    
    
    
# image_path = r'E:\cat.jpg'
# a = test_per_image(image_path)

#image_dir = '/home/ct/CycleGAN/input/wreck/trainB'
image_dir = r'E:\inception_scores\patchGAN_16'
cls_pred, softmax_out = test_images(image_dir)
mean_socres, std_scores = get_inception_socres(softmax_out)
print(mean_socres, std_scores)
