#!/usr/bin/env python
from headers import *

class BaseModel():

	def __init__(self, image_size=256, num_channels=1):
		self.image_size = image_size
		self.num_channels = num_channels
		self.num_layers = 7
 
	def initialize_base_model(self, sess, model_file=None, to_train=None, name_scope=None):

		# Initializing the session.
		self.sess = sess
		self.to_train = to_train
		
		if name_scope:
			self.name_scope = name_scope

			with tf.variable_scope(self.name_scope):
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
				
				# Define fully connected layers. 
				self.fc6_shape = 1000
				self.fc7_shape = 200

				self.fc6 = tf.layers.dense(self.flat_conv,self.fc6_shape,activation=tf.nn.relu)
				self.fc7 = tf.layers.dense(self.fc6,self.fc7_shape,activation=tf.nn.relu)
	
class ActorModel(BaseModel):

	def define_split_stream(self):

		with tf.variable_scope(self.name_scope):
			self.sigmoid_split = tf.layers.dense(self.fc7,1,activation=tf.nn.sigmoid,name='sigmoid_split')
		
			# Pixel indices, NOT normalized.
			self.lower_lim = tf.placeholder(tf.float32,shape=(None,1),name='lower_lim')
			self.upper_lim = tf.placeholder(tf.float32,shape=(None,1),name='upper_lim')

			# Predict split. 
			self.predicted_split = tf.multiply(self.upper_lim-self.lower_lim,self.sigmoid_split)+self.lower_lim

class CriticModel(BaseModel):

	def define_eval_stream(self, actor_action):

		with tf.variable_scope(self.name_scope):
			# Must take in both the image as input and the current action taken by the policy. 
			# Hence inherits from BOTH the Base and Actor Model. 
			self.fc8_shape = 40
			self.fc8 = tf.layers.dense(self.fc7,self.fc8_shape,activation=tf.nn.relu,name='fc8')

			# Concatenate the image features with the predicted split. 
			self.concat_input = tf.concat([self.fc8, actor_action],axis=-1,name='concat')

			# Now predict the Qvalue of this image state and the action. 
			self.fc9_shape = 50
			self.fc9 = tf.layers.dense(self.concat_input,self.fc9_shape,activation=tf.nn.relu,name='fc9')
			self.predicted_Qvalue = tf.layers.dense(self.fc9,1,name='predicted_Qvalue')

class ActorCriticModel():

	def __init__(self, sess, to_train=True):

		self.sess = sess
		
		# Here we instantiate the actor and critic (don't inherit).
		# with tf.name_scope('ActorModel') as scope:
		# with tf.variable_scope('ActorModel'):
		self.actor_network = ActorModel()
		self.actor_network.initialize_base_model(sess,to_train=to_train, name_scope='ActorModel')
		self.actor_network.define_split_stream()

		# with tf.name_scope('CriticModel') as scope:
		# with tf.variable_scope('CriticModel'):	
		self.critic_network = CriticModel()
		self.critic_network.initialize_base_model(sess,to_train=to_train, name_scope='CriticModel')
		# Pass the actor's action to the critic's eval function.
		self.critic_network.define_eval_stream(self.actor_network.predicted_split)	

	def define_critic_train_op(self):
		self.target_Qvalue = tf.placeholder(tf.float32, shape=(None,1), name='target_Qvalue')
		self.critic_loss = tf.losses.mean_squared_error(self.target_Qvalue, self.critic_network.predicted_Qvalue)

		# Get critic variables, to ensure gradients don't propagate through the actor.
		self.critic_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='CriticModel')

		# Creating a training operation to minimize the critic loss.
		self.critic_optimizer = tf.train.AdamOptimizer(1e-4)
		self.train_critic = self.critic_optimizer.minimize(self.critic_loss,name='Train_Critic',var_list=self.critic_variables)

	def define_actor_train_op(self):
		# Defining the actor's training op.
		self.actor_loss = -self.critic_network.predicted_Qvalue

		# Must get actor variables. 
		self.actor_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='ActorModel')
		
		self.actor_optimizer = tf.train.AdamOptimizer(1e-4)
		self.train_actor = self.actor_optimizer.minimize(self.actor_loss,name='Train_Actor',var_list=self.actor_variables)

	def define_training_ops(self):

		self.define_critic_train_op()
		self.define_actor_train_op()

		# Writing graph and other summaries in tensorflow.
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
			self.model_load(pretrained_weight_file)