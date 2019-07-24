import tensorflow as tf
import numpy as np

from Define import *

init_fn = tf.contrib.layers.xavier_initializer()
D_init_fn = init_fn # tf.truncated_normal_initializer(stddev=0.02)
G_init_fn = init_fn # tf.truncated_normal_initializer(stddev=0.02)

def residule_block(x, features, name='res', init_fn = G_init_fn):
    last_x = x
    
    x = tf.layers.conv2d(inputs = x, filters = features, kernel_size = [3, 3], strides = 1, padding = 'SAME', kernel_initializer = init_fn, name = name + '_conv_1')
    x = tf.contrib.layers.instance_norm(x, scope = name + '_instance_norm_1')
    x = tf.nn.relu(x, name = name + '_relu_1')
    
    x = tf.layers.conv2d(inputs = x, filters = features, kernel_size = [3, 3], strides = 1, padding = 'SAME', kernel_initializer = init_fn, name = name + '_conv_2')
    x = tf.contrib.layers.instance_norm(x, scope = name + '_instance_norm_2')
    x = tf.nn.relu(last_x + x, name = name + '_relu_2')
    
    return x

def Generator(inputs, reuse = False, name = 'Generator'):
    with tf.variable_scope(name, reuse = reuse):
        x = inputs

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES, kernel_size = [7, 7], strides = 1, padding = 'SAME', kernel_initializer = G_init_fn, name = 'conv_1')
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_1')
        x = tf.nn.relu(x, name = 'relu_1')
        conv_1 = x
        print(x)

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 2, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = G_init_fn, name = 'conv_2')
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_2')
        x = tf.nn.relu(x, name = 'relu_2')
        conv_2 = x
        print(x)

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 4, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = G_init_fn, name = 'conv_3')
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_3')
        x = tf.nn.relu(x, name = 'relu_3')
        conv_3 = x
        print(x)

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 8, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = G_init_fn, name = 'conv_4')
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_4')
        x = tf.nn.relu(x, name = 'relu_4')
        print(x)

        # Deconv
        x = tf.layers.conv2d_transpose(inputs = x, filters = DEFAULT_FEATURES * 2, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = G_init_fn, name = 'deconv_1')
        x = tf.contrib.layers.instance_norm(x, scope = 'deconv_instance_norm_1')
        x = tf.nn.relu(x, name = 'deconv_relu_1')
        print(x)

        x = tf.concat((x, conv_3), axis = -1)
        x = tf.layers.conv2d_transpose(inputs = x, filters = DEFAULT_FEATURES, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = G_init_fn, name = 'deconv_2')
        x = tf.contrib.layers.instance_norm(x, scope = 'deconv_instance_norm_2')
        x = tf.nn.relu(x, name = 'deconv_relu_2')
        print(x)

        x = tf.concat((x, conv_2), axis = -1)
        x = tf.layers.conv2d_transpose(inputs = x, filters = DEFAULT_FEATURES, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = G_init_fn, name = 'deconv_3')
        x = tf.contrib.layers.instance_norm(x, scope = 'deconv_instance_norm_3')
        x = tf.nn.relu(x, name = 'deconv_relu_3')
        print(x)

        # Output
        x = tf.concat((x, conv_1), axis = -1)
        x = tf.layers.conv2d(inputs = x, filters = OUTPUT_IMAGE_CHANNEL, kernel_size = [7, 7], strides = 1, padding = 'SAME', kernel_initializer = G_init_fn, name = 'last_conv')
        x = tf.nn.tanh(x, name = name + '_tanh')
        print(x)

    return x

def Discriminator(inputs, cond_inputs, reuse = False, name = 'Discriminator'):
    with tf.variable_scope(name, reuse = reuse):
        x = tf.concat((inputs, cond_inputs), axis = -1)

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = D_init_fn, name = 'conv_1') # (128, 128, ...)
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_1')
        x = tf.nn.leaky_relu(x, name = 'leaky_relu_1')

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 2, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = D_init_fn, name = 'conv_2') # (64, 64, ...)
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_2')
        x = tf.nn.leaky_relu(x, name = 'leaky_relu_2')
        
        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 4, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = D_init_fn, name = 'conv_3') # (32, 32, ...)
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_3')
        x = tf.nn.leaky_relu(x, name = 'leaky_relu_3')

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 8, kernel_size = [3, 3], strides = 1, padding = 'SAME', kernel_initializer = D_init_fn, name = 'conv_4') # (32, 32, ...)
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_4')
        x = tf.nn.leaky_relu(x, name = 'leaky_relu_4')

        x = tf.layers.conv2d(inputs = x, filters = 1, kernel_size = [3, 3], strides = 1, padding = 'SAME', kernel_initializer = D_init_fn, name = 'conv_5') # (32, 32, 1)
        #x = tf.nn.sigmoid(x)

    return x

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_IMAGE_CHANNEL])
    Generator(input_var)