# -*- coding: utf-8 -*-
"""
Created on April 8 2020

@author: Junchao Zhang

"""
import tensorflow as tf
import model as model
import os
import numpy as np
import h5py
import data_augmentation as DA

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
MAX_EPOCH = 100

MODEL_SAVE_PATH = './model-729/'
MODEL_NAME = 'HSI'

IMG_SIZE = (40, 40)
IMG_CHANNEL = 2

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    window = fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2

    v1 = 2*mu1_mu2+C1
    v2 = mu1_sq+mu2_sq+C1

    value = (v1*(2.0*sigma12 + C2))/(v2*(sigma1_sq + sigma2_sq + C2))
    value = tf.reduce_mean(value)
    value = 1.0-value
    return value


def loss_func(y_,y,qe):
    tmp_ = tf.matmul(y_,qe)
    tmp = tf.matmul(y,qe)
    mae_loss1 = tf.reduce_mean(tf.abs(tmp - tmp_))

    shape = y_.get_shape().as_list()
    yflat_ = tf.reshape(y_,[shape[0], -1, shape[3]])
    yflat = tf.reshape(y, [shape[0], -1, shape[3]])
    yflat_t = tf.transpose(yflat,perm=[0,2,1])
    yflat_t_ = tf.transpose(yflat_, perm=[0, 2, 1])
    tmp1 = tf.matmul(yflat_t,yflat)/(shape[1]*shape[2])
    tmp2 = tf.matmul(yflat_t_, yflat_)/(shape[1]*shape[2])

    loss_gram = tf.reduce_mean(tf.abs(tmp1-tmp2))


    loss_list = []
    for i in range(31):
        tmp = SSIM_LOSS(y[:, :, :, i:i + 1],y_[:, :, :, i:i + 1])
        loss_list.append(tmp)
    loss2 = tf.reduce_mean(loss_list)
    mae_loss = tf.reduce_mean(tf.abs(y - y_))
    loss = loss2 + mae_loss + mae_loss1+loss_gram
    return loss


def backward(train_data, Labels,QE,train_num):
    with tf.Graph().as_default() as g:
        with tf.name_scope('input'):
            x_rgb = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 3])
            y_ = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 31])
            qe = tf.placeholder(dtype=tf.float32, shape=[31,3])

        y = model.forward(x_rgb)
        # learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                                   train_num // BATCH_SIZE,
                                                   LEARNING_RATE_DECAY, staircase=True)
        # loss function
        with tf.name_scope('loss'):
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss1 = loss_func(y_,y,qe)
            loss = loss1 + sum(reg_losses)*0.1


        # Optimizer
        with tf.name_scope('train'):
            # Adam
            optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        # Save model
        saver = tf.train.Saver(max_to_keep=100)
        epoch = 0

        config = tf.ConfigProto(log_device_placement=True)
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1].split('-')[-2])

            while epoch < MAX_EPOCH:
                max_step = train_num // BATCH_SIZE
                listtmp = np.random.permutation(train_num)
                j = 0
                for i in range(max_step):
                    file = open("loss.txt", 'a')
                    ind = listtmp[j:j + BATCH_SIZE]
                    j = j + BATCH_SIZE
                    xs = train_data[ind, :, :, :]
                    ys = Labels[ind, :, :, :]
                    mode = np.random.permutation(8)
                    xs = DA.data_augmentation(xs,mode[0])
                    ys = DA.data_augmentation(ys, mode[0])


                    _, loss_v, step = sess.run([train_op, loss, global_step], feed_dict={x_rgb: xs, y_: ys,qe:QE})
                    file.write("Epoch: %d  Step is: %d After [ %d / %d ] training,  the batch loss is %g.\n" % (
                    epoch + 1, step, i + 1, max_step, loss_v))
                    file.close()
                    # print("Epoch: %d  After [ %d / %d ] training,  the batch loss is %g." % (epoch + 1, i + 1, max_step, loss_v))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '_epoch_' + str(epoch + 1)),
                           global_step=global_step)
                epoch += 1


if __name__ == '__main__':
    data = h5py.File(
        'F:\Hyperspectral Image Recovery\Generate Training and Testing Data\QE.mat')
    QE = data['QE']
    QE_npy = np.transpose(QE)
    print(QE_npy.shape)



    data = h5py.File('F:\Hyperspectral Image Recovery\Generate Training and Testing Data\TrainingPatches_Tensorflow_newdata_418\imdb_40_64.mat')
    input_data = data["inputs"]
    input_npy = np.transpose(input_data)

    output_data = data["outputs"]
    output_npy = np.transpose(output_data)


    print(input_npy.shape)
    train_num = input_npy.shape[0]
    backward(input_npy,  output_npy, QE_npy, train_num)