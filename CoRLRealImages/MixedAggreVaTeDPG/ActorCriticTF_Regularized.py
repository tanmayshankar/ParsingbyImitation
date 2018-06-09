#!/usr/bin/env python
from headers import *

class ActorModel():

	def __init__(self, image_size=256, num_channels=3, name_scope=None):
	 # sess, model_file=None, to_train=None, name_scope=None):
		self.image_size = image_size
		self.num_channels = num_channels
		self.num_layers = 7
		if name_scope:
			self.name_scope = name_scope

	def define_base_model(self, sess, model_file=None, to_train=None):

		self.sess = sess
		self.to_train = to_train

		# Starting regularization_coeff at 0.01
		self.regularization_coeff = 0.01

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
		self.flat_conv = tf.layers.flatten(self.conv[-1],name='flat_conv')
		
		# Define fully connected layers. 
		self.fc6_shape = 1000
		self.fc7_shape = 200

		self.fc6 = tf.layers.dense(self.flat_conv,self.fc6_shape,activation=tf.nn.relu,name='base_fc6')
		self.fc7 = tf.layers.dense(self.fc6,self.fc7_shape,activation=tf.nn.relu,name='base_fc7')

	def define_split_stream(self):
		self.actor_fc8_shape = 100
		self.actor_fc8 = tf.layers.dense(self.fc7,self.actor_fc8_shape,activation=tf.nn.tanh, name='actor_fc8',
			kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization_coeff))
		self.sigmoid_split = tf.layers.dense(self.actor_fc8,1,activation=tf.nn.sigmoid,name='sigmoid_split',
			kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization_coeff))

		# Pixel indices, NOT normalized.
		self.lower_lim = tf.placeholder(tf.float32,shape=(None,1),name='lower_lim')
		self.upper_lim = tf.placeholder(tf.float32,shape=(None,1),name='upper_lim')

		self.split_weight = tf.placeholder(tf.float32,shape=(None,1))
		# To mask gradients for non-split rules.
		self.weighted_sigmoid_split = tf.multiply(self.split_weight,self.sigmoid_split)
		# Predict split. 
		self.predicted_split = tf.add(tf.multiply(self.upper_lim-self.lower_lim,self.weighted_sigmoid_split),self.lower_lim,name='predicted_split')

	def define_rule_stream(self):
		self.num_rules = 4
		self.rule_presoftmax = tf.layers.dense(self.fc7,self.num_rules)
		self.rule_mask = tf.placeholder(tf.float32,shape=(None,self.num_rules))

		self.softmax_numerator = tf.multiply(self.rule_mask,tf.exp(self.rule_presoftmax),name='softmax_numerator')
		self.softmax_denominator = tf.add(tf.reduce_sum(tf.exp(tf.multiply(self.rule_mask,self.rule_presoftmax)),axis=-1,keep_dims=True),
			tf.subtract(tf.reduce_sum(self.rule_mask,axis=-1,keep_dims=True),tf.constant(float(self.num_rules))))
		
		self.rule_probabilities = tf.divide(self.softmax_numerator,self.softmax_denominator)

		self.argmax_rules = tf.argmax(self.rule_probabilities,axis=1,output_type=tf.int32)
		# self.rangeval = tf.range(0,tf.shape(self.argmax_rules)[0],dtype=tf.int32)
		# self.stacked_indices = tf.stack([self.rangeval,self.argmax_rules],axis=1)

		self.onehot_rules = tf.one_hot(indices=self.argmax_rules,depth=self.num_rules)

		# self.selected_rule_probabilities = tf.multiply(self.onehot_rules,self.rule_probabilities)
		# # One index for batch, one index for num_rules.
		# self.select_rule_indices = tf.placeholder(tf.int32,shape=(None,2))
		# # Must feed select_rule_indices for this.
		# # Actor loss will use this - 
		# self.selected_rule_probabilities = tf.reshape( tf.gather_nd(self.rule_probabilities,self.stacked_indices), (-1,1))

		self.rule_return_weight = tf.placeholder(tf.float32,shape=(None,1),name='rule_return_weight')
		self.target_rule = tf.placeholder(tf.float32,shape=(None,self.num_rules),name='target_rule')
		self.rule_cross_entropy = tf.keras.backend.categorical_crossentropy(self.target_rule,self.rule_probabilities)
		self.rule_loss =  tf.multiply(self.rule_return_weight,tf.expand_dims(self.rule_cross_entropy,axis=-1))

	def define_model(self,sess, model_file=None, to_train=None):
		with tf.variable_scope(self.name_scope):
			self.define_base_model(sess, model_file,to_train)
			self.define_rule_stream()
			self.define_split_stream()

class CriticModel():

	def __init__(self, image_size=256, num_channels=3, name_scope=None):
	 # sess, model_file=None, to_train=None, name_scope=None):
		self.image_size = image_size
		self.num_channels = num_channels
		self.num_layers = 7
		if name_scope:
			self.name_scope = name_scope

	def define_base_model(self, sess, model_file=None, to_train=None):

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
		self.flat_conv = tf.layers.flatten(self.conv[-1],name='flat_conv')
		
		# Define fully connected layers. 
		self.fc6_shape = 1000
		self.fc7_shape = 200

		self.fc6 = tf.layers.dense(self.flat_conv,self.fc6_shape,activation=tf.nn.relu,name='base_fc6')
		self.fc7 = tf.layers.dense(self.fc6,self.fc7_shape,activation=tf.nn.relu,name='base_fc7')
	
	def define_eval_stream(self, actor_split, onehot_rules):
		# Must take in both the image as input and the current action taken by the policy. 
		# Hence inherits from BOTH the Base and Actor Model. 
		self.fc8_shape = 40
		self.fc8 = tf.layers.dense(self.fc7,self.fc8_shape,activation=tf.nn.relu,name='critic_fc8')

		# Concatenate the image features with the predicted split. 
		self.concat_input = tf.concat([self.fc8, actor_split, onehot_rules],axis=-1,name='concat')

		# Now predict the Qvalue of this image state and the action. 
		self.fc9_shape = 50
		self.fc9 = tf.layers.dense(self.concat_input,self.fc9_shape,activation=tf.nn.tanh,name='critic_fc9')
		self.predicted_Qvalue = tf.layers.dense(self.fc9,1,name='predicted_Qvalue')

	def define_model(self,sess, model_file=None, to_train=None, actor_split=None, onehot_rules=None):
		with tf.variable_scope(self.name_scope):
			self.define_base_model(sess, model_file,to_train)
			self.define_eval_stream(actor_split, onehot_rules)

class ActorCriticModel():

	def __init__(self, sess, to_train=True):

		self.sess = sess
		self.to_train = to_train		
		# Here we instantiate the actor and critic (don't inherit).

		self.actor_network = ActorModel(name_scope='ActorModel')
		self.actor_network.define_model(sess,to_train=to_train)		

		self.critic_network = CriticModel(name_scope='CriticModel')
		self.critic_network.define_model(sess,to_train=to_train, actor_split=self.actor_network.predicted_split, onehot_rules=self.actor_network.onehot_rules)

	def define_critic_train_op(self):
		self.target_Qvalue = tf.placeholder(tf.float32, shape=(None,1), name='target_Qvalue')
		self.critic_loss = tf.losses.mean_squared_error(self.target_Qvalue, self.critic_network.predicted_Qvalue)

		# Get critic variables, to ensure gradients don't propagate through the actor.
		self.critic_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='CriticModel')

		# Creating a training operation to minimize the critic loss.
		self.critic_optimizer = tf.train.AdamOptimizer(1e-4)
		# self.train_critic = self.critic_optimizer.minimize(self.critic_loss,name='Train_Critic',var_list=self.critic_variables)

		# Clipping gradients because of NaN values. 
		self.critic_gradients_vars = self.critic_optimizer.compute_gradients(self.critic_loss,var_list=self.critic_variables)
		self.critic_clipped_gradients = [(tf.clip_by_norm(grad,10),var) for grad, var in self.critic_gradients_vars]
		# self.train_critic = self.critic_optimizer.apply_gradients(self.critic_gradients_vars)
		self.train_critic = self.critic_optimizer.apply_gradients(self.critic_clipped_gradients)

	def define_actor_train_op(self):
		# Defining the actor's training op.
		# Actor split loss (from DDPG). 
		self.actor_split_loss = -self.critic_network.predicted_Qvalue+ tf.losses.get_regularization_losses(scope=self.actor_network.name_scope)
		self.actor_rule_loss = self.actor_network.rule_loss
		self.actor_loss = self.actor_rule_loss+self.actor_split_loss
		
		# Must get actor variables. 
		self.actor_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='ActorModel')
		
		self.actor_optimizer = tf.train.AdamOptimizer(1e-4)
		# self.train_actor = self.actor_optimizer.minimize(self.actor_loss,name='Train_Actor',var_list=self.actor_variables)

		# Clipping gradients because of NaN values. 
		self.actor_gradients_vars = self.actor_optimizer.compute_gradients(self.actor_loss,var_list=self.actor_variables)
		self.actor_clipped_gradients = [(tf.clip_by_norm(grad,10),var) for grad, var in self.actor_gradients_vars]
		# self.train_actor = self.actor_optimizer.apply_gradients(self.actor_gradients_vars)
		self.train_actor = self.actor_optimizer.apply_gradients(self.actor_clipped_gradients)

	def define_training_ops(self):

		self.define_critic_train_op()
		self.define_actor_train_op()

		# Writing graph and other summaries in tensorflow.
		if self.to_train:
			self.writer = tf.summary.FileWriter('training',self.sess.graph)			
		init = tf.global_variables_initializer()
		self.sess.run(init)

	def define_logging_ops(self):
		# Create file writer to write summaries. 		
		self.tf_writer = tf.summary.FileWriter('train_logging'+'/',self.sess.graph)

		# Create summaries for: Log likelihood, reward weight, and total reward on the full image. 
		self.actor_loss_summary = tf.summary.scalar('Actor_Loss',tf.reduce_mean(self.actor_loss))
		self.critic_loss_summary = tf.summary.scalar('Critic_Loss',tf.reduce_mean(self.critic_loss))

		# Merge summaries. 
		self.merged_summaries = tf.summary.merge_all()		
	
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

	def model_load_alt(self, model_file):
		print("RESTORING MODEL FROM:", model_file)
		saver = tf.train.Saver(max_to_keep=None)
		saver.restore(self.sess,model_file)

	def save_model(self, model_index, iteration_number=-1):
		if not(os.path.isdir("saved_models")):
			os.mkdir("saved_models")

		self.saver = tf.train.Saver(max_to_keep=None)           

		if not(iteration_number==-1):
			save_path = self.saver.save(self.sess,'saved_models/model_epoch{0}_iter{1}.ckpt'.format(model_index,iteration_number))
		else:
			save_path = self.saver.save(self.sess,'saved_models/model_epoch{0}.ckpt'.format(model_index))

	def create_network(self, sess, pretrained_weight_file=None, to_train=False):

		self.define_training_ops()
		self.define_logging_ops()
		
		if pretrained_weight_file:
			# self.model_load(pretrained_weight_file)
			self.model_load_alt(pretrained_weight_file)
