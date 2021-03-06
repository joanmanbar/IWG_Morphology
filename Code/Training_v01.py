#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_datasets as datasets
from skimage import io

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Images = io.ImageCollection('../Images/Stems/Train/images/*.tif')
# Labels = io.ImageCollection('../Images/Stems/Train/masks/*.tif')
# print(f"found {len(fX)} training images and {len(fY)} training masks")

# plt.imshow(fX[0], cmap='gray')
# fX[0].shape



# Loading the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print('Training Input size: ', x_train.shape)
# assert x_train.shape == (60000, 28, 28)
# assert x_test.shape == (10000, 28, 28)
# assert y_train.shape == (60000,)
# assert y_test.shape == (10000,)

# Sample image
plt.imshow(x_train[100])

# Hyper parameters
learning_rate = 0.0001
epochs = 10
batch_size = 50          # ~10% of images

# Define input-output placeholders
x = tf.placeholder(tf.float32, [None, 748])
y = tf.placeholder(tf.float32, [None, 10])

# Reshape the input into a tensor
x_shaped = tf.reshape(x, [-1, 28, 28, 1])     #[batch size, nrow, ncol,  nchannel]



def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    
    # Initialize weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    
    # Setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    
    # Add the bias
    out_layer += bias
    
    # Apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)
    
    # Now perform max pooling-reduce the feature map size
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    
    return out_layer



# Create some convolutional layers
layer1 = create_new_conv_layer(x_shaped, 1, 32, [5,5], [2,2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5,5], [2,2], name='layer2')

# For fully connected layers
flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

# Setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

# Another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)


# Define loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

# Add an optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# Create session and start training
with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    total_batch = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = y_train.next_batch.next_batch(batch_size=batch_size)
            _, c = sess.run([optimizer, cross_entropy],
                            feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
        print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost),
              "test accuracy: {:.3f}".format(test_acc))
        
    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: x_test, y: y_test}))




