#!/usr/bin/env python
import os
import scipy as scp
import scipy.misc

import numpy as np
import numpy as npy
import logging
import tensorflow as tf
import sys

# import fcn16_vgg
# import modified_gradient_network
import fcn_pretrain_FCGRAD
import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
					level=logging.INFO,
					stream=sys.stdout)

from tensorflow.python.framework import ops

# img1 = scp.misc.imread("../test_data/tabby_cat.png")
input_images = npy.load("IMAGES_TO_USE.npy")
grads = npy.load("SOFTMAX_GRADS.npy")


num_epochs = 20
num_images = 276
image_size = 256
# Add 0's to the end of every gradient value. 
image_gradients = npy.zeros((num_images,2,image_size))
for i in range(num_images):
	image_gradients[i,0,:-1] = grads[i][0]
	image_gradients[i,1,:-1] = grads[i][1]

loss = npy.zeros(num_images)

# # Create a TensorFlow session with limits on GPU usage.
gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list="3,2")
config = tf.ConfigProto(gpu_options=gpu_ops)
# sess = tf.Session(config=config)

with tf.Session(config=config) as sess:

	# images = tf.placeholder(tf.float32, shape=(256,256,3))
	# batch_images = tf.expand_dims(images, 0)

	# vgg_fcn = fcn16_vgg.FCN16VGG()
	# vgg_fcn = modified_gradient_network.FCN16VGG()
	vgg_fcn = fcn_pretrain_FCGRAD.FCN16VGG()

	# Building with 2 classes, so that we can sum horizontally and vertically for gradients across each class.
	with tf.name_scope("content_vgg"):
		vgg_fcn.build(train=True, num_classes=2)

	print('Finished building Network.')

	init = tf.global_variables_initializer()
	sess.run(init)

	print('Running the Network')


	for e in range(num_epochs):

		z = range(num_images)
		npy.random.shuffle(z)
		
		loss = npy.zeros(num_images)
		hgrad = npy.zeros((num_images,image_size))
		vgrad = npy.zeros((num_images,image_size))

		for i in range(num_images):
			print("Epoch:",e,"Image:",i)
			index = z.pop()

			feed_dict = {vgg_fcn.input_image: input_images[index], vgg_fcn.horizontal_image_gradients: image_gradients[index,1], vgg_fcn.vertical_image_gradients: image_gradients[index,0]}
			_, loss[index], hgrad[index], vgrad[index] = sess.run([vgg_fcn.train, vgg_fcn.total_loss, vgg_fcn.horizontal_grad, vgg_fcn.vertical_grad], feed_dict=feed_dict)

		print("AFTER EPOCH",e,"Loss value:",loss.mean())
		npy.save("HGrad_Epoch{0}.npy".format(e),hgrad)
		npy.save("VGrad_Epoch{0}.npy".format(e),vgrad)
	# tensors = [vgg_fcn.horizontal_grad, vgg_fcn.vertical_grad]
	# h, v = sess.run(tensors,feed_dict=feed_dict)	


