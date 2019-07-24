import tensorflow as tf

from Define import *

def L1_Loss(outputs, targets):
    return tf.reduce_mean(tf.abs(outputs - targets))

def Pixel2Pixel_Loss(dst_input_var, dst_image_op, D_fake_op, D_real_op):
    D_fake_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_op, labels = tf.zeros_like(D_fake_op)))
    D_real_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_op, labels = tf.ones_like(D_real_op)))
    D_loss_op = D_real_loss_op + D_fake_loss_op

    G_fake_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_op, labels = tf.ones_like(D_fake_op)))
    G_style_loss_op = LOSS_LAMBDA * L1_Loss(dst_input_var, dst_image_op)
    G_loss_op = G_fake_loss_op + G_style_loss_op

    return G_loss_op, G_fake_loss_op, G_style_loss_op, D_loss_op

