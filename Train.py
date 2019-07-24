import os
import cv2
import sys
import glob
import time

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *
from Pixel2Pixel import *
from Pixel2Pixel_Loss import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
SRC_IMAGE_DIRS = ['C:/DB/CMP_facade_DB/image/']

train_image_paths = []
for image_dir in SRC_IMAGE_DIRS:
    train_image_paths += glob.glob(image_dir + '*')

np.random.shuffle(train_image_paths)
train_length = len(train_image_paths)

valid_image_paths = train_image_paths[:int(train_length * 0.1)]
train_image_paths = train_image_paths[int(train_length * 0.1):]

# 2. build
src_input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, OUTPUT_IMAGE_CHANNEL])
dst_input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_IMAGE_CHANNEL])

dst_image_op = Generator(src_input_var, reuse = False)

D_fake_op = Discriminator(dst_image_op, src_input_var, reuse = False)
D_real_op = Discriminator(dst_input_var, src_input_var, reuse = True)

# 3. loss
G_loss_op, G_fake_loss_op, G_style_loss_op, D_loss_op = Pixel2Pixel_Loss(dst_input_var, dst_image_op, D_fake_op, D_real_op)

# 4. select variables
vars = tf.trainable_variables()
D_vars = [var for var in vars if 'Discriminator' in var.name]
G_vars = [var for var in vars if 'Generator' in var.name]

print('[i] Discriminator')
for var in D_vars:
    print(var)

print('[i] Generator')
for var in G_vars:
    print(var)

# 5. optimizer
learning_rate_var = tf.placeholder(tf.float32, name = 'learning_rate')
D_train_op = tf.train.AdamOptimizer(learning_rate_var, beta1 = 0.5).minimize(D_loss_op, var_list = D_vars)
G_train_op = tf.train.AdamOptimizer(learning_rate_var, beta1 = 0.5).minimize(G_loss_op, var_list = G_vars)

# 6. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
#saver.restore(sess, './model/Pixel2Pixel_{}.ckpt'.format())

learning_rate = INIT_LEARNING_RATE
train_iteration = len(train_image_paths) // BATCH_SIZE

for epoch in range(MAX_EPOCH):

    train_time = time.time()
    D_loss_list = []
    G_loss_list = []
    G_fake_loss_list = []
    G_style_loss_list = []

    np.random.shuffle(train_image_paths)
    for iter in range(train_iteration):
        batch_src_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_IMAGE_CHANNEL), dtype = np.float32)
        batch_dst_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, OUTPUT_IMAGE_CHANNEL), dtype = np.float32)

        batch_image_paths = train_image_paths[iter * BATCH_SIZE : (iter + 1) * BATCH_SIZE]
        for index, image_path in enumerate(batch_image_paths):
            src_image, dst_image = Get_Domain_Data(image_path, 'facade')
            
            batch_src_image_data[index] = src_image.astype(np.float32) / 127.5 - 1
            batch_dst_image_data[index] = dst_image.astype(np.float32) / 127.5 - 1

        _feed_dict = {src_input_var : batch_src_image_data, dst_input_var : batch_dst_image_data, learning_rate_var : learning_rate}

        _, G_loss, G_fake_loss, G_style_loss = sess.run([G_train_op, G_loss_op, G_fake_loss_op, G_style_loss_op], feed_dict = _feed_dict)
        G_loss_list.append(G_loss)
        G_fake_loss_list.append(G_fake_loss)
        G_style_loss_list.append(G_style_loss)

        _, D_loss = sess.run([D_train_op, D_loss_op], feed_dict = _feed_dict)
        D_loss_list.append(D_loss)

        sys.stdout.write('\r[{}/{}] D_loss : {:.5f}, G_loss : {:.5f}, G_fake_loss : {:.5f}, G_style_loss : {:.5f}'.format(iter, train_iteration, D_loss, G_loss, G_fake_loss, G_style_loss))
        sys.stdout.flush()

    train_time = int(time.time() - train_time)
    D_loss = np.mean(D_loss_list)
    G_loss = np.mean(G_loss_list)
    G_fake_loss = np.mean(G_fake_loss_list)
    G_style_loss = np.mean(G_style_loss_list)

    print()
    log_print("[i] epoch : {}, D_loss : {:.5f}, G_loss : {:.5f}, G_fake_loss : {:.5f}, G_style_loss : {:.5f}, time : {}sec".format(epoch, D_loss, G_loss, G_fake_loss, G_style_loss, train_time))

    fake_images = np.zeros((SAVE_WIDTH * SAVE_HEIGHT, IMAGE_HEIGHT, IMAGE_WIDTH, OUTPUT_IMAGE_CHANNEL), dtype = np.float32)

    for i in range(SAVE_WIDTH):
        src_image, dst_image = Get_Domain_Data(valid_image_paths[i], DOMAIN_OPTION)

        fake_images[SAVE_WIDTH * 0 + i] = src_image.astype(np.float32) / 127.5 - 1
        fake_images[SAVE_WIDTH * 1 + i] = sess.run(dst_image_op, feed_dict = {src_input_var : [src_image]})[0]

    for i in range(SAVE_WIDTH):
        src_image, dst_image = Get_Domain_Data(valid_image_paths[SAVE_WIDTH + i], DOMAIN_OPTION)

        fake_images[SAVE_WIDTH * 2 + i] = src_image.astype(np.float32) / 127.5 - 1
        fake_images[SAVE_WIDTH * 3 + i] = sess.run(dst_image_op, feed_dict = {src_input_var : [src_image]})[0]

    Save(fake_images, './results/{}.jpg'.format(epoch))
    saver.save(sess, './model/Pixel2Pixel_{}.ckpt'.format(epoch))
