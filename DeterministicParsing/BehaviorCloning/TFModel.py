#!/usr/bin/env python
from headers import *

class Model():

	def __init__(self, image_size=256, num_channels=1):
		self.image_size = image_size
		self.num_channels = num_channels
		self.num_layers = 7

	def initialize_base_model(self, sess, model_file=None, to_train=None):

		# Initializing the session.
		self.sess = sess
		self.to_train = to_train

		# Number of layers. 
		self.num_fc_layers = 2
		self.conv_sizes = 3*npy.ones((self.num_layers),dtype=int)		
		self.conv_num_filters = 20*npy.ones((self.num_layers),dtype=int)
		self.conv_strides = 2*npy.ones((self.num_layers),dtype=int)

		self.conv_strides[0:3] = 1
		# # Strides are now: 1,1,1,2,2,2,2

		# Placeholders
		# Now doing this for single channel images.
		self.input = tf.placeholder(tf.float32,shape=[None,self.image_size,self.image_size,self.num_channels],name='input')

		# Defining conv layers.
		self.conv = [[] for i in range(self.num_layers)]

		# Initial layer.
		self.conv[0] = tf.layers.conv2d(self.input,filters=self.conv_num_filters[0],kernel_size=(self.conv_sizes[0]),strides=(self.conv_strides[0]),activation=tf.nn.relu,name='conv0')

		# Defining subsequent conv layers.
		for i in range(1,self.num_layers):
			self.conv[i] = tf.layers.conv2d(self.conv[i-1],filters=self.conv_num_filters[i],kernel_size=(self.conv_sizes[i]),strides=(self.conv_strides[i]),activation=tf.nn.relu,name='conv{0}'.format(i))

		# Now going to flatten this and move to a fully connected layer. 		
		self.flat_conv = tf.layers.flatten(self.conv[-1])

	def define_rule_stream(self):
		self.rule_fc6_shape = 1000
		self.rule_fc7_shape = 200
		self.rule_fc6 = tf.layers.dense(self.flat_conv,self.rule_fc6_shape,activation=tf.nn.relu)
		self.rule_fc7 = tf.layers.dense(self.rule_fc6,self.rule_fc7_shape,activation=tf.nn.relu)

		self.num_rules = 4
		self.rule_presoftmax = tf.layers.dense(self.rule_fc7,self.num_rules)
		self.rule_mask = tf.placeholder(tf.float32,shape=(None,self.num_rules))

		self.softmax_numerator = tf.multiply(self.rule_mask,tf.exp(self.rule_presoftmax),name='softmax_numerator')
		self.softmax_denominator = tf.add(tf.reduce_sum(tf.exp(tf.multiply(self.rule_mask,self.rule_presoftmax)),axis=-1,keep_dims=True),
			tf.subtract(tf.reduce_sum(self.rule_mask,axis=-1,keep_dims=True),tf.constant(float(self.num_rules))))
		
		self.rule_probabilities = tf.divide(self.softmax_numerator,self.softmax_denominator)

		self.rule_return_weight = tf.placeholder(tf.float32,shape=(None,1),name='rule_return_weight')
		self.target_rule = tf.placeholder(tf.float32,shape=(None,self.num_rules),name='target_rule')

		# self.target_rule = tf.placeholder(tf.float32,shape=(None,self.num_rules),name='target_rule')
		self.rule_cross_entropy = tf.keras.backend.categorical_crossentropy(self.target_rule,self.rule_probabilities)
		# self.rule_loss = tf.multiply(self.rule_return_weight,self.rule_cross_entropy)
		self.rule_loss =  tf.multiply(self.rule_return_weight,tf.expand_dims(self.rule_cross_entropy,axis=-1))

	def define_split_stream(self):
		self.fc6_shape = 1000
		self.fc7_shape = 200

		self.fc6 = tf.layers.dense(self.flat_conv,self.fc6_shape,activation=tf.nn.relu)
		self.fc7 = tf.layers.dense(self.fc6,self.fc7_shape,activation=tf.nn.relu)
		self.sigmoid_split = tf.layers.dense(self.fc7,1,activation=tf.nn.sigmoid)
	
		# Pixel indices, NOT normalized.
		self.lower_lim = tf.placeholder(tf.float32,shape=(None,1),name='lower_lim')
		self.upper_lim = tf.placeholder(tf.float32,shape=(None,1),name='upper_lim')

		self.predicted_split = tf.multiply(self.upper_lim-self.lower_lim,self.sigmoid_split)+self.lower_lim

		self.target_split = tf.placeholder(tf.float32,shape=(None,1),name='target_split')
		self.split_mse = tf.losses.mean_squared_error(self.target_split,self.predicted_split)
		self.split_return_weight = tf.placeholder(tf.float32,shape=(None,1),name='split_return_weight')
		self.split_loss = tf.multiply(self.split_mse,self.split_return_weight)

	def logging_ops(self):
		# Create file writer to write summaries. 		
		self.tf_writer = tf.summary.FileWriter('train_logging'+'/',self.sess.graph)

		# Create summaries for: Log likelihood, reward weight, and total reward on the full image. 
		# self.split_loglikelihood_summary = tf.summary.scalar('Split_LogLikelihood',tf.reduce_mean(self.split_loglikelihood))
		self.rule_loglikelihood_summary = tf.summary.scalar('Rule_LogLikelihood',tf.reduce_mean(self.rule_cross_entropy))
		self.reward_weight_summary = tf.summary.scalar('Reward_Weight',tf.reduce_mean(self.rule_return_weight))
		# self.split_mean_summary = tf.summary.scalar('Split_Mean',tf.reduce_mean(self.normal_mean))
		# self.split_var_summary = tf.summary.scalar('Split_Var',tf.reduce_mean(self.normal_var))		

		# Merge summaries. 
		self.merged_summaries = tf.summary.merge_all()		

	def training_ops(self):

		self.split_loss_lambda = tf.constant(1.)

		self.total_loss = self.rule_loss+tf.multiply(self.split_loss_lambda,self.split_loss)
		# self.total_loss = self.split_loss

		# Creating a training operation to minimize the total loss.
		self.optimizer = tf.train.AdamOptimizer(1e-4)
		self.train = self.optimizer.minimize(self.total_loss,name='Adam_Optimizer')

		# Writing graph and other summaries in tensorflow.
		self.writer = tf.summary.FileWriter('training',self.sess.graph)
		init = tf.global_variables_initializer()
		self.sess.run(init)
	
	def model_load(self, model_file):
		# DEFINING CUSTOM LOADER:
		print("RESTORING MODEL FROM:", model_file)
		reader = tf.train.NewCheckpointReader(model_file)
		saved_shapes = reader.get_variable_to_shape_map()
		var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
			if var.name.split(':')[0] in saved_shapes])
		restore_vars = []
		name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
		with tf.variable_scope('', reuse=True):
			for var_name, saved_var_name in var_names:
				curr_var = name2var[saved_var_name]
				var_shape = curr_var.get_shape().as_list()
				if var_shape == saved_shapes[saved_var_name]:
					restore_vars.append(curr_var)
		saver = tf.train.Saver(max_to_keep=None,var_list=restore_vars)
		saver.restore(self.sess, model_file)

	def save_model(self, model_index, iteration_number=-1):
		if not(os.path.isdir("saved_models")):
			os.mkdir("saved_models")

		self.saver = tf.train.Saver(max_to_keep=None)           

		if not(iteration_number==-1):
			save_path = self.saver.save(self.sess,'saved_models/model_epoch{0}_iter{1}.ckpt'.format(model_index,iteration_number))
		else:
			save_path = self.saver.save(self.sess,'saved_models/model_epoch{0}.ckpt'.format(model_index))

	def create_network(self, sess, pretrained_weight_file=None, to_train=False):

		self.initialize_base_model(sess,to_train=to_train)
		self.define_rule_stream()
		self.define_split_stream()
		self.logging_ops()
		self.training_ops()

		if pretrained_weight_file:
			self.model_load(pretrained_weight_file)