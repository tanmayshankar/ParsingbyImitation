#!/usr/bin/env python
import numpy as npy
import tensorflow as tf


# # Create a TensorFlow session with limits on GPU usage.
gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list="3,2")
config = tf.ConfigProto(gpu_options=gpu_ops)
# sess = tf.Session(config=config)

num_epochs = 50
num_images = 276
image_size = 256

# img1 = scp.misc.imread("../test_data/tabby_cat.png")
input_images = npy.load("IMAGES_TO_USE.npy")
grads = npy.load("SOFTMAX_GRADS.npy")

# Add 0's to the end of every gradient value. 
image_gradients = npy.zeros((num_images,2,image_size))
for i in range(num_images):
	image_gradients[i,0,:-1] = grads[i][0]
	image_gradients[i,1,:-1] = grads[i][1]

loss = npy.zeros(num_images)


with tf.Session(config=config) as sess:

	input_image = tf.placeholder(tf.float32,shape=(1,image_size,image_size,3))
	horz_grad = tf.placeholder(tf.float32,shape=(image_size))
	vert_grad = tf.placeholder(tf.float32,shape=(image_size))

	num_layers = 5
	W_conv = [[] for i in range(num_layers)]
	b_conv = [[] for i in range(num_layers)]

	conv = [[] for i in range(num_layers)]
	relu_conv = [[] for i in range(num_layers)]
	conv_num_filters = npy.array([3,20,20,20,20,20],dtype=int)
	conv_sizes = 3*npy.ones((num_layers),dtype=int)		
	conv_strides = npy.array([1,2,2,2,2])

	for i in range(num_layers):
		W_conv[i] = tf.Variable(tf.truncated_normal([conv_sizes[i],conv_sizes[i],conv_num_filters[i],conv_num_filters[i+1]],stddev=0.1),name='W_conv{0}'.format(i+1))
		b_conv[i] = tf.Variable(tf.constant(0.1,shape=[conv_num_filters[i+1]]),name='b_conv{0}'.format(i+1))    

	conv[0] = tf.add(tf.nn.conv2d(input_image,W_conv[0],strides=[1,conv_strides[0],conv_strides[0],1],padding='VALID'),b_conv[0],name='conv0')	
	relu_conv[0] = tf.nn.relu(conv[0],name='relu0')

	# Defining subsequent conv layers.
	for i in range(1,num_layers):
		conv[i] = tf.add(tf.nn.conv2d(conv[i-1],W_conv[i],strides=[1,conv_strides[i],conv_strides[i],1],padding='VALID'),b_conv[i],name='conv{0}'.format(i+1))
		relu_conv[i] = tf.nn.relu(conv[i],name='relu{0}'.format(i+1))

	fc_input_shape = 14*14*conv_num_filters[-1]
	fc_input = tf.reshape(relu_conv[-1],[-1,fc_input_shape],name='fc_input')

	fc1_num_hidden = 600
	fc2_num_hidden = 400
	fc_output = image_size*2

	W_fc1 = tf.Variable(tf.truncated_normal([fc_input_shape,fc1_num_hidden],stddev=0.1),name='W_fc1')
	b_fc1 = tf.Variable(tf.constant(0.1,shape=[fc1_num_hidden]))
	W_fc2 = tf.Variable(tf.truncated_normal([fc1_num_hidden,fc2_num_hidden],stddev=0.1),name='W_fc2')
	b_fc2 = tf.Variable(tf.constant(0.1,shape=[fc2_num_hidden]))
	W_fc3 = tf.Variable(tf.truncated_normal([fc2_num_hidden,fc_output],stddev=0.1),name='W_fc3')
	b_fc3 = tf.Variable(tf.constant(0.1,shape=[fc_output]))

	fc1 = tf.nn.relu(tf.matmul(fc_input,W_fc1)+b_fc1)
	fc2 = tf.nn.relu(tf.matmul(fc1,W_fc2)+b_fc2)
	fc3 = tf.nn.relu(tf.matmul(fc2,W_fc3)+b_fc3)
	print(fc3)
	hgrad_presf = fc3[0][:image_size]
	print(hgrad_presf)
	vgrad_presf = fc3[0][image_size:]
	print(vgrad_presf)
	hgrad = tf.nn.softmax(hgrad_presf)
	vgrad = tf.nn.softmax(vgrad_presf)

	hloss = tf.nn.softmax_cross_entropy_with_logits(labels=horz_grad,logits=hgrad_presf)
	vloss = tf.nn.softmax_cross_entropy_with_logits(labels=vert_grad,logits=vgrad_presf)

	total_loss = hloss+vloss

	optimizer = tf.train.AdamOptimizer(1e-4)
	train = optimizer.minimize(total_loss,name='Adam_Optimizer')

	sess.run(tf.global_variables_initializer())

	print('Running the Network')
	for e in range(num_epochs):
		z = range(num_images)
		npy.random.shuffle(z)
		loss = npy.zeros(num_images)
		thgrad = npy.zeros((num_images,image_size))
		tvgrad = npy.zeros((num_images,image_size))

		for i in range(num_images):
			print("Epoch:",e,"Image:",i)
			index = z.pop()

			feed_dict = {input_image: input_images[index].reshape(1,256,256,3), horz_grad: image_gradients[index,1], vert_grad: image_gradients[index,0]}
			_, loss[index], thgrad[index], tvgrad[index] = sess.run([train, total_loss, hgrad, vgrad], feed_dict=feed_dict)

		print("AFTER EPOCH",e,"Loss value:",loss.mean())
		npy.save("HGrad_Epoch{0}.npy".format(e),thgrad)
		npy.save("VGrad_Epoch{0}.npy".format(e),tvgrad)
	# tensors = [vgg_fcn.horizontal_grad, vgg_fcn.vertical_grad]
	# h, v = sess.run(tensors,feed_dict=feed_dict)	











