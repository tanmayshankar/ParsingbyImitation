#!/usr/bin/env python
from headers import *
from state_class import *

class hierarchical():

	def __init__(self):

		self.num_epochs = 10
		self.save_every = 100
		self.num_images = 276
		self.current_parsing_index = 0
		self.parse_tree = [parse_tree_node()]
		self.paintwidth = -1
		self.minimum_width = -1
		self.images = []
		self.original_images = []
		self.true_labels = []
		self.image_size = -1
		self.intermittent_lambda = 0.
		self.suffix = []

#####################################################################################################
#####################################################################################################
	
	# REMEMBER, FIRST BUILDING THE NETWORK FOR RESIZED 256x256 image input.
	def build(self, sess, model_file=None):
		self.sess = sess

		# if vgg16_npy_path is None:
		path = sys.modules[self.__class__.__module__].__file__
		path = os.path.abspath(os.path.join(path, os.pardir))
		path = os.path.join(path, "vgg16.npy")
		vgg16_npy_path = path
		# logging.info("Load npy file from '%s'.", vgg16_npy_path)

		# if not os.path.isfile(vgg16_npy_path):
		# 	logging.error(("File '%s' not found. Download it from "
		# 				   "ftp://mi.eng.cam.ac.uk/pub/mttt2/"
		# 				   "models/vgg16.npy"), vgg16_npy_path)
		# 	sys.exit(1)
		debug = False
		random_init_fc8 = False
		VGG_MEAN = [ 175.5183833 ,  176.6830765 ,  192.35719172]
		self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
		self.wd = 5e-4
		print("npy file loaded")

		"""
		Build the VGG model using loaded weights
		Parameters
		----------
		rgb: image batch tensor
			Image in rgb shap. Scaled to Intervall [0, 255]
		train: bool
			Whether to build train or inference graph
		num_classes: int
			How many classes should be predicted (by fc8)
		random_init_fc8 : bool
			Whether to initialize fc8 layer randomly.
			Finetuning is required in this case.
		debug: bool
			Whether to print additional Debug Information.
		"""
		train = True
		num_classes = 2
		# with tf.device('/device:GPU:0'):
		with tf.device('/gpu:0'):
			# self.input = tf.placeholder(tf.float32,shape=(1,256,256,3))
			self.input = tf.placeholder(tf.float32,shape=(None,None,3))
			self.expanded_input = tf.expand_dims(self.input,0)
			red, green, blue = tf.split(self.expanded_input, 3, 3)
			# red, green, blue = tf.split(self.input, 3, 3)

			self.bgr = tf.concat([blue - VGG_MEAN[0],green - VGG_MEAN[1],red - VGG_MEAN[2]], axis=3)
			self.conv1_1 = self._conv_layer(self.bgr, "conv1_1")
			self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
			self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

			self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
			self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
			self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

			self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
			self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
			self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
			self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)
		with tf.device('/gpu:1'):
			self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
			self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
			self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
			self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

			self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
			self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")

			self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
			self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)
		with tf.device('/gpu:2'):

			self.fc6 = self._fc_layer(self.pool5, "fc6")

			if train:
				self.fc6 = tf.nn.dropout(self.fc6, 0.5)

			self.fc7 = self._fc_layer(self.fc6, "fc7")
			if train:
				self.fc7 = tf.nn.dropout(self.fc7, 0.5)

			# For 256x256 images (for which we resize to), FC7 always takes shape 8x8x4096. 
			# There is no need to do the global spatial pooling to obtain constant shape. 
			# We can get away by just reshaping. 
			# When we switch to NOT resizing the original input, for an FCN,
			# we can do global spatial pooling (averaging) to feed into the rule / primitive streams. 

			# self.fc_input_shape = 8*8*4096	
			# self.policy_branch_fcinput = tf.reshape(self.fc7,[-1,self.fc_input_shape])

		with tf.device('/gpu:0'):
			# # if random_init_fc8:
			# # 	self.score_fr = self._score_layer(self.fc7, "score_fr", num_classes)
			# # else:
			# # 	self.score_fr = self._fc_layer(self.fc7, "score_fr", num_classes=num_classes, relu=False)

			# # self.fc_input_shape = 8*8*num_classes	
			# # self.policy_branch_fcinput = tf.reshape(self.score_fr,[-1,self.fc_input_shape])

			# # self.pred = tf.argmax(self.score_fr, dimension=3)
			# # self.upscore2 = self._upscore_layer(self.score_fr, shape=tf.shape(self.pool4), num_classes=num_classes, debug=debug, name='upscore2', ksize=4, stride=2)

			# # From 4096 to 128. 
			# # self.feature_dimension = 64			
			# # self.num_filters = 64
			# # self.finalconv_output = self._fc_layer(self.fc7,"finalconv_output",num_classes=self.feature_dimension, relu=True)
			# ###########################################################################################
			# # Putting a fully-convolutional layer here. 
			# self.W_fullyconv = tf.Variable(tf.truncated_normal([1,1,4096,64],stddev=0.1),name='W_fullyconv')
			# self.b_fullyconv = tf.Variable(tf.constant(0.1,shape=[64]),name='b_fullyconv')

			# self.finalconv_output = tf.nn.relu(tf.nn.conv2d(self.fc7,self.W_fullyconv,strides=[1,1,1,1],padding='SAME')+self.b_fullyconv,name='finalconv_output')
			# ###########################################################################################


			self.score_fr = self._fc_layer(self.fc7, "score_fr", num_classes=num_classes, relu=False)

			self.fc_input_shape = 4096
			# self.policy_branch_fcinput = tf.reshape(self.finalconv_output,[-1,self.fc_input_shape])
			self.policy_branch_fcinput = tf.reduce_mean(self.fc7,axis=[1,2])

			self.pred = tf.argmax(self.score_fr, dimension=3)
			self.upscore2 = self._upscore_layer(self.score_fr, shape=tf.shape(self.pool4), num_classes=num_classes, debug=debug, name='upscore2', ksize=4, stride=2)


		with tf.device('/gpu:1'):
			self.score_pool4 = self._score_layer(self.pool4, "score_pool4", num_classes=num_classes)
			self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)
			self.upscore32 = self._upscore_layer(self.fuse_pool4, shape=tf.shape(self.bgr), num_classes=num_classes, debug=debug, name='upscore32', ksize=32, stride=16)
			self.pred_up = tf.argmax(self.upscore32, dimension=3)

			self.horizontal_grad_presf = tf.reduce_sum(self.upscore32[:,:,:,0],axis=1)
			self.vertical_grad_presf = tf.reduce_sum(self.upscore32[:,:,:,1],axis=2)

			self.horizontal_grad = tf.nn.softmax(self.horizontal_grad_presf)
			self.vertical_grad = tf.nn.softmax(self.vertical_grad_presf)

		################################################################################################

		########## COMMON FC LAYERS ####################################################################
		# with tf.device('/device:GPU:1'):
		with tf.device('/gpu:2'):

			# Rule stream
			self.rule_num_fclayers = 3
			self.rule_num_hidden1 = 1000
			self.rule_num_hidden2 = 400
			self.rule_num_branches = 4
			self.rule_fc_shapes = [[self.fc_input_shape,self.rule_num_hidden1,self.rule_num_hidden2,6],
								   [self.fc_input_shape,self.rule_num_hidden1,self.rule_num_hidden2,4],
								   [self.fc_input_shape,self.rule_num_hidden1,self.rule_num_hidden2,4],
								   [self.fc_input_shape,self.rule_num_hidden1,self.rule_num_hidden2,2]]
			
			# Split stream
			self.split_num_branches = 2

			# Primitive stream
			self.primitive_num_fclayers = 3
			self.primitive_num_hidden1 = 1000
			self.primitive_num_hidden2 = 400
			self.number_primitives = 4
			self.primitive_fc_shapes = [self.fc_input_shape,self.primitive_num_hidden1,self.primitive_num_hidden2,self.number_primitives]

			# Reshape FC input.
			self.target_rule_shapes = [6,4,4,2]	

			################################################################################################

			########## RULE FC LAYERS ######################################################################
			
			# Defining FC layer variables lists.
			self.W_rule_fc = [[[] for i in range(self.rule_num_fclayers)] for j in range(self.rule_num_branches)]
			self.b_rule_fc = [[[] for i in range(self.rule_num_fclayers)] for j in range(self.rule_num_branches)]

			# Defining rule_fc layers.
			self.rule_fc = [[[] for i in range(self.rule_num_fclayers)] for j in range(self.rule_num_branches)]
			self.rule_probabilities = [[] for j in range(self.rule_num_branches)]
			# self.rule_dist = [[] for j in range(self.rule_num_branches)]

			# Can maintain a single sample_rule and sampled_rule for all of the branches, because we will use tf.case for each.
			self.rule_indicator = tf.placeholder(tf.int32,name='rule_indicator')
			self.sampled_rule = tf.placeholder(tf.int32,name='sampled_rule')

			# Defining Rule FC variables.
			for j in range(self.rule_num_branches):
				# for i in range(self.num_fc_layers):
				for i in range(self.rule_num_fclayers):
					self.W_rule_fc[j][i] = tf.Variable(tf.truncated_normal([self.rule_fc_shapes[j][i],self.rule_fc_shapes[j][i+1]],stddev=0.1),name='W_rulefc_branch{0}_layer{1}'.format(j,i+1))
					self.b_rule_fc[j][i] = tf.Variable(tf.constant(0.1,shape=[self.rule_fc_shapes[j][i+1]]),name='b_rulefc_branch{0}_layer{1}'.format(j,i+1))
				# self.sampled_rule[j] = tf.placeholder(tf.int32)
			# Defining Rule FC layers.
			for j in range(self.rule_num_branches):
				# self.rule_fc[j][0] = tf.nn.relu(tf.add(tf.matmul(self.fc_input,self.W_rule_fc[j][0]),self.b_rule_fc[j][0]),name='rule_fc_branch{0}_layer0'.format(j))
				self.rule_fc[j][0] = tf.nn.relu(tf.add(tf.matmul(self.policy_branch_fcinput,self.W_rule_fc[j][0]),self.b_rule_fc[j][0]),name='rule_fc_branch{0}_layer0'.format(j))
				self.rule_fc[j][1] = tf.nn.relu(tf.add(tf.matmul(self.rule_fc[j][0],self.W_rule_fc[j][1]),self.b_rule_fc[j][1],name='rule_fc_branch{0}_layer1'.format(j)))
				self.rule_fc[j][2] = tf.add(tf.matmul(self.rule_fc[j][1],self.W_rule_fc[j][2]),self.b_rule_fc[j][2],name='rule_fc_branch{0}_layer2'.format(j))
				self.rule_probabilities[j] = tf.nn.softmax(self.rule_fc[j][2],name='rule_probabilities_branch{0}'.format(j))
				
				# Now not using the categorical distributions, directly taking the rule_probabilities so we can sample greedily at test time, or do stuff like epsilon greedy.
				# self.rule_dist[j] = tf.contrib.distributions.Categorical(probs=self.rule_probabilities[j],name='rule_dist_branch{0}'.format(j))
				# self.sample_rule[j] = self.rule_dist[j].sample()
			
			# This is the heart of the routing. We select which set of parameters we sample the rule from.
			# Default needs to be a lambda function because we're providing arguments to it. 
			# self.sample_rule = tf.case({tf.equal(self.rule_indicator,0): self.rule_dist[0].sample, tf.equal(self.rule_indicator,1): self.rule_dist[1].sample, 
			# 							tf.equal(self.rule_indicator,2): self.rule_dist[2].sample, tf.equal(self.rule_indicator,3): self.rule_dist[3].sample},default=lambda: -tf.ones(1),exclusive=True,name='sample_rule')

			# self.selected_rule_probabilities = tf.case({tf.equal(self.rule_indicator,0): self.rule_probabilities[0], tf.equal(self.rule_indicator,1): self.rule_probabilities[1], 
										# tf.equal(self.rule_indicator,2): self.rule_probabilities[2], tf.equal(self.rule_indicator,3): self.rule_probabilities[3]},default=lambda: -tf.zeros(1),exclusive=True,name='selected_rule_probabilities')

			self.selected_rule_probabilities = tf.case({tf.equal(self.rule_indicator,0): lambda: self.rule_probabilities[0], tf.equal(self.rule_indicator,1): lambda: self.rule_probabilities[1], 
										tf.equal(self.rule_indicator,2): lambda: self.rule_probabilities[2], tf.equal(self.rule_indicator,3): lambda: self.rule_probabilities[3]},default=lambda: -tf.zeros(1),exclusive=True,name='selected_rule_probabilities')

			################################################################################################

			########### SPLIT FC LAYERS ####################################################################
		
			self.split_dist = [[] for j in range(self.split_num_branches)]

			# Similarly to rules, we can use one sample_split, because we use tf.case.
			self.split_indicator = tf.placeholder(tf.int32,name='split_indicator')
			# self.sampled_split = tf.placeholder(tf.float32,name='sampled_split')
			self.sampled_split = tf.placeholder(tf.int32,name='sampled_split')

			# Defining distributions for each.

			# self.split_dist[0] = tf.contrib.distributions.Normal(loc=self.split_mean[j],scale=self.split_cov[j],name='split_dist_branch{0}'.format(j))
			self.split_dist[0] = tf.contrib.distributions.Categorical(probs=self.horizontal_grad)
			self.split_dist[1] = tf.contrib.distributions.Categorical(probs=self.vertical_grad)
				# self.sample_split[j] = self.split_dist[j].sample()

			# This is the heart of the routing. We select which set of parameters we sample the split location from. 
			# Default needs to be a lambda function because we're providing arguments to it. 		
			# self.sample_split = tf.case({tf.equal(self.split_indicator,0): self.split_dist[0].sample,tf.equal(self.split_indicator,1): self.split_dist[1].sample},default=lambda: -tf.ones(1),exclusive=True,name='sample_split')

			################################################################################################

			########## PRIMITIVE FC LAYERS #################################################################

			# Defining FC layaer for primitive stream.
			self.W_primitive_fc = [[] for i in range(self.primitive_num_fclayers)]
			self.b_primitive_fc = [[] for i in range(self.primitive_num_fclayers)]

			# Defining primitive fc layers.
			self.primitive_fc = [[] for i in range(self.primitive_num_fclayers)]

			# Defining variables:
			for i in range(self.primitive_num_fclayers):
				self.W_primitive_fc[i] = tf.Variable(tf.truncated_normal([self.primitive_fc_shapes[i],self.primitive_fc_shapes[i+1]],stddev=0.1),name='W_primitivefc_layer{0}'.format(i+1))
				self.b_primitive_fc[i] = tf.Variable(tf.constant(0.1,shape=[self.primitive_fc_shapes[i+1]]),name='b_primitivefc_layer{0}'.format(i+1))
			
			# Defining primitive FC layers.
			self.primitive_fc[0] = tf.nn.relu(tf.add(tf.matmul(self.policy_branch_fcinput,self.W_primitive_fc[0]),self.b_primitive_fc[0]),name='primitve_fc_layer0')		
			self.primitive_fc[1] = tf.add(tf.matmul(self.primitive_fc[0],self.W_primitive_fc[1]),self.b_primitive_fc[1],name='primitve_fc_layer1')
			self.primitive_fc[2] = tf.add(tf.matmul(self.primitive_fc[1],self.W_primitive_fc[2]),self.b_primitive_fc[2],name='primitve_fc_layer2')
			self.primitive_probabilities = tf.nn.softmax(self.primitive_fc[-1],name='primitive_probabilities')
			
			# Defining categorical distribution for primitives.
			# self.primitive_dist = tf.contrib.distributions.Categorical(probs=self.primitive_probabilities,name='primitive_distribution')
			
			################################################################################################

			########### NOW MOVING TO THE LOSS FUNCTIONS ###################################################

			self.return_weight = tf.placeholder(tf.float32,name='return_weight')

			######## For rule stream:##########
			self.target_rule = [[] for j in range(self.rule_num_branches)] 
			self.rule_loss_branch = [[] for j in range(self.rule_num_branches)]

			# Defining a log probability loss for each of the rule policy branches.
			for j in range(self.rule_num_branches):
				self.target_rule[j] = tf.placeholder(tf.float32,shape=(self.rule_fc_shapes[j][-1]))

				# print(self.target_rule[j].shape,self.rule_fc_shapes[j][-1].shape)
				# print(self.target_rule[j],self.rule_fc[j][-1])
				# self.rule_loss_branch[j] = tf.multiply(self.return_weight,tf.nn.softmax_cross_entropy_with_logits(labels=self.target_rule[j],logits=self.rule_fc[j][-1]),name='rule_loss_branch{0}'.format(j))			
				self.rule_loss_branch[j] = tf.nn.softmax_cross_entropy_with_logits(labels=self.target_rule[j],logits=self.rule_fc[j][-1],name='rule_loss_branch{0}'.format(j))

			# Defining a loss that selects which branch to back-propagate into.
			self.rule_loss = tf.case({tf.equal(self.rule_indicator,0): lambda: self.rule_loss_branch[0], tf.equal(self.rule_indicator,1): lambda: self.rule_loss_branch[1], 
										tf.equal(self.rule_indicator,2): lambda: self.rule_loss_branch[2], tf.equal(self.rule_indicator,3): lambda: self.rule_loss_branch[3]},default=lambda: tf.zeros(1),exclusive=True,name='rule_loss')

			######## For split stream:#########
			self.split_loss_branch = [[] for j in range(self.split_num_branches)]

			for j in range(self.split_num_branches):
				# self.split_loss_branch = -tf.multply(self.return_weight,self.split_dist[j].log_prob(self.sampled_split),name='split_loss_branch{0}'.format(j))
				self.split_loss_branch[j] = -self.split_dist[j].log_prob(self.sampled_split)

			# Now defining a split loss that selects which branch to back-propagate into.
			# self.split_loss = tf.case({tf.equal(self.split_indicator,0): self.split_dist[0].sample,tf.equal(self.split_indicator,1): self.split_dist[1].sample},default=lambda: tf.zeros(1,dtype=tf.int32),exclusive=True,name='sample_split')
			self.split_loss = tf.case({tf.equal(self.split_indicator,0): lambda: self.split_loss_branch[0],tf.equal(self.split_indicator,1): lambda: self.split_loss_branch[j]},default=lambda: tf.zeros(1),exclusive=True,name='sample_split')
			######### For primitive stream ####

			self.target_primitive = tf.placeholder(tf.float32,shape=(self.number_primitives))
			self.primitive_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.target_primitive,logits=self.primitive_fc[-1],name='primitive_loss')

			######### COMBINED LOSS ###########

			self.policy_indicator = tf.placeholder(tf.int32)
			# There is a fixed set of policy branches that can be applied together. 
			# A split rule is applied with the split location --> Case 1
			# An assignment rule is run on its own --> Case 2
			# A primitive is run on its own --> Case 3
			
			self.selected_loss = tf.case({tf.equal(self.policy_indicator,0): lambda: self.rule_loss+self.split_loss, tf.equal(self.policy_indicator,1): lambda: self.rule_loss, tf.equal(self.policy_indicator,2): lambda: self.primitive_loss},default=lambda: tf.zeros(1), exclusive=True,name='selected_loss')
			self.total_loss = tf.multiply(self.return_weight,self.selected_loss,name='total_loss')

			#################################################################################################
		self.previous_goal = npy.zeros(2)
		self.current_start = npy.zeros(2)

		# Defining the training optimizer. 
		self.optimizer = tf.train.AdamOptimizer(1e-4)
		self.train = self.optimizer.minimize(self.total_loss,name='Adam_Optimizer')

		# Writing graph and other summaries in tensorflow.
		self.writer = tf.summary.FileWriter('training',self.sess.graph)
		init = tf.global_variables_initializer()
		self.sess.run(init)

		#################################
		if model_file:
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
		#################################

		# Maintaining list of all goals and start locations. 
		self.goal_list = []
		self.start_list = []

######################################################################################################
######################################################################################################

######################################################################################################
# UTILITY CODE FOR BUILDING AND LOADING VGG WEIGHTS.
######################################################################################################

	def _max_pool(self, bottom, name, debug):
		pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
							  padding='SAME', name=name)

		if debug:
			pool = tf.Print(pool, [tf.shape(pool)],
							message='Shape of %s' % name,
							summarize=4, first_n=1)
		return pool

	def _conv_layer(self, bottom, name):
		with tf.variable_scope(name) as scope:
			filt = self.get_conv_filter(name)
			conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

			conv_biases = self.get_bias(name)
			bias = tf.nn.bias_add(conv, conv_biases)

			relu = tf.nn.relu(bias)
			# Add summary to Tensorboard
			_activation_summary(relu)
			return relu

	def _fc_layer(self, bottom, name, num_classes=None, relu=True, debug=False):
		with tf.variable_scope(name) as scope:
			shape = bottom.get_shape().as_list()

			if name == 'fc6':
				filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
			elif name == 'score_fr':
				name = 'fc8'  # Name of score_fr layer in VGG Model
				filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
												  num_classes=num_classes)
			else:
				filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])
			conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
			conv_biases = self.get_bias(name, num_classes=num_classes)
			bias = tf.nn.bias_add(conv, conv_biases)

			if relu:
				bias = tf.nn.relu(bias)
			_activation_summary(bias)

			if debug:
				bias = tf.Print(bias, [tf.shape(bias)],
								message='Shape of %s' % name,
								summarize=4, first_n=1)
			return bias

	def _score_layer(self, bottom, name, num_classes):
		with tf.variable_scope(name) as scope:
			# get number of input channels
			in_features = bottom.get_shape()[3].value
			shape = [1, 1, in_features, num_classes]
			# He initialization Sheme
			if name == "score_fr":
				num_input = in_features
				stddev = (2 / num_input)**0.5
			elif name == "score_pool4":
				stddev = 0.001
			# Apply convolution
			w_decay = self.wd
			weights = self._variable_with_weight_decay(shape, stddev, w_decay)
			conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
			# Apply bias
			conv_biases = self._bias_variable([num_classes], constant=0.0)
			bias = tf.nn.bias_add(conv, conv_biases)

			_activation_summary(bias)

			return bias

	def _upscore_layer(self, bottom, shape, num_classes, name, debug, ksize=4, stride=2):
		strides = [1, stride, stride, 1]
		with tf.variable_scope(name):
			in_features = bottom.get_shape()[3].value

			if shape is None:
				# Compute shape out of Bottom
				in_shape = tf.shape(bottom)

				h = ((in_shape[1] - 1) * stride) + 1
				w = ((in_shape[2] - 1) * stride) + 1
				new_shape = [in_shape[0], h, w, num_classes]
			else:
				new_shape = [shape[0], shape[1], shape[2], num_classes]
			output_shape = tf.stack(new_shape)

			# logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
			f_shape = [ksize, ksize, num_classes, in_features]

			# create
			num_input = ksize * ksize * in_features / stride
			stddev = (2 / num_input)**0.5

			weights = self.get_deconv_filter(f_shape)
			deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
											strides=strides, padding='SAME')

			if debug:
				deconv = tf.Print(deconv, [tf.shape(deconv)],
								  message='Shape of %s' % name,
								  summarize=4, first_n=1)

		_activation_summary(deconv)
		return deconv

	def get_deconv_filter(self, f_shape):
		width = f_shape[0]
		heigh = f_shape[0]
		f = ceil(width/2.0)
		c = (2 * f - 1 - f % 2) / (2.0 * f)
		bilinear = np.zeros([f_shape[0], f_shape[1]])
		for x in range(width):
			for y in range(heigh):
				value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
				bilinear[x, y] = value
		weights = np.zeros(f_shape)
		for i in range(f_shape[2]):
			weights[:, :, i, i] = bilinear

		init = tf.constant_initializer(value=weights,
									   dtype=tf.float32)
		return tf.get_variable(name="up_filter", initializer=init,
							   shape=weights.shape)

	def get_conv_filter(self, name):
		init = tf.constant_initializer(value=self.data_dict[name][0],
									   dtype=tf.float32)
		shape = self.data_dict[name][0].shape
		print('Layer name: %s' % name)
		print('Layer shape: %s' % str(shape))
		var = tf.get_variable(name="filter", initializer=init, shape=shape)
		if not tf.get_variable_scope().reuse:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
									   name='weight_loss')
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
								 weight_decay)
		return var

	def get_bias(self, name, num_classes=None):
		bias_wights = self.data_dict[name][1]
		shape = self.data_dict[name][1].shape
		if name == 'fc8':
			bias_wights = self._bias_reshape(bias_wights, shape[0],
											 num_classes)
			shape = [num_classes]
		init = tf.constant_initializer(value=bias_wights,
									   dtype=tf.float32)
		return tf.get_variable(name="biases", initializer=init, shape=shape)

	def get_fc_weight(self, name):
		init = tf.constant_initializer(value=self.data_dict[name][0],
									   dtype=tf.float32)
		shape = self.data_dict[name][0].shape
		var = tf.get_variable(name="weights", initializer=init, shape=shape)
		if not tf.get_variable_scope().reuse:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
									   name='weight_loss')
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
								 weight_decay)
		return var

	def _bias_reshape(self, bweight, num_orig, num_new):
		""" Build bias weights for filter produces with `_summary_reshape`

		"""
		n_averaged_elements = num_orig//num_new
		avg_bweight = np.zeros(num_new)
		for i in range(0, num_orig, n_averaged_elements):
			start_idx = i
			end_idx = start_idx + n_averaged_elements
			avg_idx = start_idx//n_averaged_elements
			if avg_idx == num_new:
				break
			avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
		return avg_bweight

	def _summary_reshape(self, fweight, shape, num_new):
		""" Produce weights for a reduced fully-connected layer.

		FC8 of VGG produces 1000 classes. Most semantic segmentation
		task require much less classes. This reshapes the original weights
		to be used in a fully-convolutional layer which produces num_new
		classes. To archive this the average (mean) of n adjanced classes is
		taken.

		Consider reordering fweight, to perserve semantic meaning of the
		weights.

		Args:
		  fweight: original weights
		  shape: shape of the desired fully-convolutional layer
		  num_new: number of new classes


		Returns:
		  Filter weights for `num_new` classes.
		"""
		num_orig = shape[3]
		shape[3] = num_new
		assert(num_new < num_orig)
		n_averaged_elements = num_orig//num_new
		avg_fweight = np.zeros(shape)
		for i in range(0, num_orig, n_averaged_elements):
			start_idx = i
			end_idx = start_idx + n_averaged_elements
			avg_idx = start_idx//n_averaged_elements
			if avg_idx == num_new:
				break
			avg_fweight[:, :, :, avg_idx] = np.mean(
				fweight[:, :, :, start_idx:end_idx], axis=3)
		return avg_fweight

	def _variable_with_weight_decay(self, shape, stddev, wd):
		"""Helper to create an initialized Variable with weight decay.

		Note that the Variable is initialized with a truncated normal
		distribution.
		A weight decay is added only if one is specified.

		Args:
		  name: name of the variable
		  shape: list of ints
		  stddev: standard deviation of a truncated Gaussian
		  wd: add L2Loss weight decay multiplied by this float. If None, weight
			  decay is not added for this Variable.

		Returns:
		  Variable Tensor
		"""

		initializer = tf.truncated_normal_initializer(stddev=stddev)
		var = tf.get_variable('weights', shape=shape,
							  initializer=initializer)

		if wd and (not tf.get_variable_scope().reuse):
			weight_decay = tf.multiply(
				tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
								 weight_decay)
		return var

	def _bias_variable(self, shape, constant=0.0):
		initializer = tf.constant_initializer(constant)
		return tf.get_variable(name='biases', shape=shape,
							   initializer=initializer)

	def get_fc_weight_reshape(self, name, shape, num_classes=None):
		print('Layer name: %s' % name)
		print('Layer shape: %s' % shape)
		weights = self.data_dict[name][0]
		weights = weights.reshape(shape)
		if num_classes is not None:
			weights = self._summary_reshape(weights, shape,
											num_new=num_classes)
		init = tf.constant_initializer(value=weights,
									   dtype=tf.float32)
		return tf.get_variable(name="weights", initializer=init, shape=shape)

######################################################################################################
######################################################################################################

	def save_model(self, model_index, iteration_number=-1):
		if not(os.path.isdir("saved_models")):
			os.mkdir("saved_models")

		self.saver = tf.train.Saver(max_to_keep=None)           

		if not(iteration_number==-1):
			save_path = self.saver.save(self.sess,'saved_models/model_epoch{0}_iter{1}.ckpt'.format(model_index,iteration_number))
		else:
			save_path = self.saver.save(self.sess,'saved_models/model_epoch{0}.ckpt'.format(model_index))

	def initialize_tree(self):
		# Intialize the parse tree for this image.=
		self.state = parse_tree_node(label=0,x=0,y=0,w=self.image_size,h=self.image_size)
		self.current_parsing_index = 0
		self.parse_tree = [parse_tree_node()]
		self.parse_tree[self.current_parsing_index]=self.state

	def insert_node(self, state, index):    
		self.parse_tree.insert(index,state)

	def remap_rule_indices(self, rule_index):

		if self.state.rule_indicator==0:
			# Remember, allowing all 6 rules.
			return rule_index
		elif self.state.rule_indicator==1: 
			# Now allowing only vertical splits and assignments. 
			if rule_index>=2:
				return rule_index+2
			else:
				return rule_index*2
		elif self.state.rule_indicator==2:
			# Now allowing only horizontal splits and assignments.
			if rule_index>=2:
				return rule_index+2
			else:
				return rule_index*2+1
		elif self.state.rule_indicator==3:
			# Now allowing only assignment rules.
			return rule_index+4

	def set_rule_indicator(self):
		if self.state.h<=self.minimum_width and self.state.w<=self.minimum_width:
			# Allowing only assignment.
			self.state.rule_indicator = 3
		elif self.state.h<=self.minimum_width:
			# Allowing only horizontal splits and assignment.
			self.state.rule_indicator = 2
		elif self.state.w<=self.minimum_width:
			# Allowing only vertical splits and assignment.
			self.state.rule_indicator = 1
		else:
			# Allowing anything and everything.
			self.state.rule_indicator = 0

# May need to rewrite parse_nonterminal
	def parse_nonterminal(self, image_index):

		# Four branches of the rule policy.
		self.set_rule_indicator()

		# rule_probabilities = self.sess.run(self.selected_rule_probabilities, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,3),  self.rule_indicator: self.state.rule_indicator})
		rule_probabilities = self.sess.run(self.selected_rule_probabilities, feed_dict={self.input: self.resized_image, self.rule_indicator: self.state.rule_indicator})
		# Must handle the fact that branches now index rules differently, using remap_rule_indices.
		if self.to_train:
			selected_rule = npy.random.choice(range(len(rule_probabilities[0])),p=rule_probabilities[0])
		elif not(self.to_train):
			selected_rule = npy.argmax(rule_probabilities[0])

		self.parse_tree[self.current_parsing_index].rule_applied = copy.deepcopy(selected_rule)
		# print("PARSING:")
		# print(self.state.rule_indicator)
		# print("Selected rule:",selected_rule)

		selected_rule = self.remap_rule_indices(selected_rule)
		indices = self.map_rules_to_indices(selected_rule)
		split_location = -1

		#####################################################################################
		# Split rule selected.
		if selected_rule<=3:

			# Resampling until it gets a split INSIDE the segment. This just ensures the split lies within 0 and 1.
			if ((selected_rule==0) or (selected_rule==2)):
				counter = 0             
				self.state.split_indicator = 0
				# SAMPLING SPLIT LOCATION INSIDE THIS CONDITION:

				while (split_location<=0)or(split_location>=self.state.h):
					# probs = self.sess.run(self.horizontal_grad, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,3), self.split_indicator: self.state.split_indicator})	
					# probs = self.sess.run(self.horizontal_grad, feed_dict={self.input: self.resized_image, self.split_indicator: self.state.split_indicator})	
					probs = copy.deepcopy(self.horizontal_grad)
					epsilon = 0.0001
					categorical_prob_softmax = copy.deepcopy(probs)
					# categorical_prob_softmax += epsilon
					categorical_prob_softmax[[0,-1]] = 0.
					categorical_prob_softmax = categorical_prob_softmax/categorical_prob_softmax.sum()
					# split_location = npy.random.choice(range(self.image_size),p=categorical_prob_softmax)
					split_location = npy.random.choice(range(len(categorical_prob_softmax)),p=categorical_prob_softmax)
					print("HEEYYYY")
					print(categorical_prob_softmax.shape)	
					# print(categorical_prob_softmax)	
					# if split_location>=((self.uy-self.ly)/2):
						# split_location = int(npy.floor(float(self.state.h*split_location)/self.image_size))
					# else:
						# split_location = int(npy.ceil(float(self.state.h*split_location)/self.image_size))
					
					# print(counter)			
					counter+=1
				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=self.state.w,h=split_location,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x,y=self.state.y+split_location,w=self.state.w,h=self.state.h-split_location,backward_index=self.current_parsing_index)

			if ((selected_rule==1) or (selected_rule==3)):
				counter = 0
				self.state.split_indicator = 1

				while (split_location<=0)or(split_location>=self.state.w):
					# probs = self.sess.run(self.horizontal_grad, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,3), self.split_indicator: self.state.split_indicator})	
					# probs = self.sess.run(self.horizontal_grad, feed_dict={self.input: self.resized_image, self.split_indicator: self.state.split_indicator})	
					probs = copy.deepcopy(self.vertical_grad)

					epsilon = 0.0001
					categorical_prob_softmax = copy.deepcopy(probs)
					# categorical_prob_softmax += epsilon
					categorical_prob_softmax[[0,-1]] = 0.
					categorical_prob_softmax = categorical_prob_softmax/categorical_prob_softmax.sum()
					# split_location = npy.random.choice(range(self.image_size),p=categorical_prob_softmax)
					split_location = npy.random.choice(range(len(categorical_prob_softmax)),p=categorical_prob_softmax)
					print("HEEYYYY")
					print(categorical_prob_softmax.shape)	
					# print(categorical_prob_softmax)	
					# if split_location>=((self.ux-self.lx)/2):
					# 	split_location = int(npy.floor(float(self.state.w*split_location)/self.image_size))
					# else:
					# 	split_location = int(npy.ceil(float(self.state.w*split_location)/self.image_size))
					
					print(counter)			
					counter+=1
				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=split_location,h=self.state.h,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x+split_location,y=self.state.y,w=self.state.w-split_location,h=self.state.h,backward_index=self.current_parsing_index)
				
			# Update current parse tree with split location and rule applied.
			# self.parse_tree[self.current_parsing_index].split = split_copy
			self.parse_tree[self.current_parsing_index].boundaryscaled_split = split_location
			# self.parse_tree[self.current_parsing_index].rule_applied = selected_rule
			self.parse_tree[self.current_parsing_index].alter_rule_applied = selected_rule

			self.predicted_labels[image_index,s1.x:s1.x+s1.w,s1.y:s1.y+s1.h] = s1.label
			self.predicted_labels[image_index,s2.x:s2.x+s2.w,s2.y:s2.y+s2.h] = s2.label
			
			if (selected_rule<=1):
				# Insert splits into parse tree.
				self.insert_node(s1,self.current_parsing_index+1)
				self.insert_node(s2,self.current_parsing_index+2)

			if (selected_rule>=2):
				# Insert splits into parse tree.
				self.insert_node(s2,self.current_parsing_index+1)
				self.insert_node(s1,self.current_parsing_index+2)

			self.current_parsing_index+=1
		
		#####################################################################################
		# Assignment rules:
		elif selected_rule>=4:
			
			# Now even with the different primitives we don't need more than 6 rules; since choice of primitive is independent of assignment of primitive.
			# Create a parse tree node object.
			s1 = copy.deepcopy(self.parse_tree[self.current_parsing_index])
			# Change label.
			s1.label=selected_rule-3
			# Change the backward index (backwardly linked link list)
			s1.backward_index = self.current_parsing_index

			# Update current parse tree with rule applied.
			# self.parse_tree[self.current_parsing_index].rule_applied = selected_rule
			self.parse_tree[self.current_parsing_index].alter_rule_applied = selected_rule

			# Insert node into parse tree.
			self.insert_node(s1,self.current_parsing_index+1)
			self.current_parsing_index+=1                       
			self.predicted_labels[image_index,s1.x:s1.x+s1.w,s1.y:s1.y+s1.h] = s1.label         

# May not need to rewrite parse_primitive_terminal
	def parse_primitive_terminal(self, image_index):
		# Sample a goal location.
		self.nonpaint_moving_term = 0.
		self.strokelength_term = 0.

		# If it is a region to be painted and assigned a primitive:
		if (self.state.label==1):

			# primitive_probabilities = self.sess.run(self.primitive_probabilities, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,3)})              
			primitive_probabilities = self.sess.run(self.primitive_probabilities, feed_dict={self.input: self.resized_image})              

			if self.to_train:
				selected_primitive = npy.random.choice(range(self.number_primitives),p=primitive_probabilities[0])
			if not(self.to_train):
				selected_primitive = npy.argmax(primitive_probabilities[0])

			# For primitive 0, horizontal brush stroke from left to right. (at bottom)
			# For primitive 1, horizontal brush stroke from right to left. (at bottom)
			# For primitive 2, vertical brush stroke from top to bottom. (at left)
			# For primitive 3, vertical brush stroke from bottom to top. (at left)
			# print("Selected Primitive:",selected_primitive)
			
			if (selected_primitive==0):
				self.current_start = npy.array([self.state.y+self.state.h/2,self.state.x])
				self.current_goal = npy.array([self.state.y+self.state.h/2,self.state.x+self.state.w])

				lower = max(self.state.y,self.state.y+(self.state.h-self.paintwidth)/2)
				upper = min(self.state.y+(self.state.h+self.paintwidth)/2,self.state.y+self.state.h)

				self.painted_image[self.state.x:(self.state.x+self.state.w), lower:upper] = 1.
				self.painted_images[image_index, self.state.x:(self.state.x+self.state.w), lower:upper] = 1.

			if (selected_primitive==1):
				self.current_start = npy.array([self.state.y+self.state.h/2,self.state.x+self.state.w])
				self.current_goal = npy.array([self.state.y+self.state.h/2,self.state.x])

				lower = max(self.state.y,self.state.y+(self.state.h-self.paintwidth)/2)
				upper = min(self.state.y+(self.state.h+self.paintwidth)/2,self.state.y+self.state.h)

				self.painted_image[self.state.x:(self.state.x+self.state.w), lower:upper] = 1.
				self.painted_images[image_index, self.state.x:(self.state.x+self.state.w), lower:upper] = 1.

			if (selected_primitive==2):
				self.current_start = npy.array([self.state.y,self.state.x+self.state.w/2])
				self.current_goal = npy.array([self.state.y+self.state.h,self.state.x+self.state.w/2])              

				lower = max(self.state.x,self.state.x+(self.state.w-self.paintwidth)/2)             
				upper = min(self.state.x+(self.state.w+self.paintwidth)/2,self.state.x+self.state.w)

				self.painted_image[lower:upper, self.state.y:self.state.y+self.state.h] = 1.
				self.painted_images[image_index,lower:upper, self.state.y:self.state.y+self.state.h] = 1.

			if (selected_primitive==3):
				self.current_start = npy.array([self.state.y+self.state.h,self.state.x+self.state.w/2])
				self.current_goal = npy.array([self.state.y,self.state.x+self.state.w/2])
				
				lower = max(self.state.x,self.state.x+(self.state.w-self.paintwidth)/2)             
				upper = min(self.state.x+(self.state.w+self.paintwidth)/2,self.state.x+self.state.w)

				self.painted_image[lower:upper, self.state.y:self.state.y+self.state.h] = 1.
				self.painted_images[image_index,lower:upper, self.state.y:self.state.y+self.state.h] = 1.

			self.nonpaint_moving_term = npy.linalg.norm(self.current_start-self.previous_goal)**2/((self.image_size)**2)
			self.strokelength_term = npy.linalg.norm(self.current_goal-self.current_start)**2/((self.image_size)**2)

			self.previous_goal = copy.deepcopy(self.current_goal)

			self.start_list.append(self.current_start)
			self.goal_list.append(self.current_goal)

			self.parse_tree[self.current_parsing_index].primitive = selected_primitive

		self.state.reward = (self.true_labels[image_index, self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]*self.painted_image[self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]).sum()

		self.state.stroke_term = copy.deepcopy(self.strokelength_term)
		self.state.intermittent_term = copy.deepcopy(self.nonpaint_moving_term)

		self.current_parsing_index+=1

	def propagate_rewards(self):

		# Traverse the tree in reverse order, accumulate rewards into parent nodes recursively as sum of rewards of children.
		# This is actually the return accumulated by any particular decision.
		# Now we are discounting based on the depth of the tree (not just sequence in episode)
		self.gamma = 1.

		for j in reversed(range(len(self.parse_tree))): 
			if (self.parse_tree[j].backward_index>=0):
				self.parse_tree[self.parse_tree[j].backward_index].reward += self.parse_tree[j].reward*self.gamma

		for j in range(len(self.parse_tree)):
			self.parse_tree[j].reward /= (self.parse_tree[j].w*self.parse_tree[j].h)            
		self.alpha = 1.0
		
		# Non-linearizing rewards.
		for j in range(len(self.parse_tree)):
			self.parse_tree[j].reward = npy.tan(self.alpha*self.parse_tree[j].reward)       

		# Additional term for continuity. 
		for j in range(len(self.parse_tree)):
			self.parse_tree[j].reward -= self.parse_tree[j].intermittent_term*self.intermittent_lambda
			# print("MODIFIED REWARD:",self.parse_tree[j].reward)

# Definitely need to rewrite backprop.
	def backprop(self, image_index):
		# Must decide whether to do this stochastically or in batches. # For now, do it stochastically, moving forwards through the tree.
		for j in range(len(self.parse_tree)):
			self.state = self.parse_tree[j]
			
			boundary_width = 0
			lowerx = max(0,self.state.x-boundary_width)
			upperx = min(self.image_size,self.state.x+self.state.w+boundary_width)
			lowery = max(0,self.state.y-boundary_width)
			uppery = min(self.image_size,self.state.y+self.state.h+boundary_width)

			self.image_input = self.images[image_index, lowerx:upperx, lowery:uppery, :]
			self.resized_image = copy.deepcopy(self.image_input)
			# self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))

			# Must set indicator functions.
			policy_indicator = -1
			rule_indicator = -1
			split_indicator = -1

			# Declare target rule and primitives.
			# target_rule = npy.zeros(self.fcs1_output_shape)
			target_rule = [[] for i in range(self.rule_num_branches)]

			for k in range(self.rule_num_branches):
				target_rule[k] = npy.zeros(self.target_rule_shapes[k])
			# target_rule = npy.zeros(self.target_rule_shapes[self.parse_tree[j].rule_indicator])                       
			target_primitive = npy.zeros(self.number_primitives)        

			# Set the return weight for the loss globally.
			return_weight = self.parse_tree[j].reward

			# Here, we set the indicator functions for the various cases.
			# If it's a non-terminal:
			if self.parse_tree[j].label==0:         
				# A rule was necessarily applied, so set the target rule and the rule branch indicator.
				# target_rule[self.parse_tree[j].rule_applied] = 1.
				# print("Ind:")
				# print(self.parse_tree[j].rule_indicator)
				# print(self.parse_tree[j].rule_applied)

				target_rule[self.parse_tree[j].rule_indicator][self.parse_tree[j].rule_applied] = 1.
				rule_indicator = self.parse_tree[j].rule_indicator
				# If it was a split rule, then both the split and rule policies were used. 
				if self.parse_tree[j].rule_applied<=3:
					policy_indicator = 0
					# Since a split location was sampled, provide the split indicator for branch.
					split_indicator = self.parse_tree[j].split_indicator
				else:
					policy_indicator = 1                    

			# If it was a terminal symbol that was to be painted:
			if self.parse_tree[j].label==1:
				# Set the target primitive and policy branch.
				target_primitive[self.parse_tree[j].primitive] = 1.             
				policy_indicator = 2

			# Remember, we don't backprop for a terminal not to be painted (since we already would've backpropagated gradients
			# for assigning the parent non-terminal to a region not to be painted).

			# self.sess.run(self.train, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,3), #self.sampled_split: int(self.parse_tree[j].split), \
			self.sess.run(self.train, feed_dict={self.input: self.resized_image, self.sampled_split: self.parse_tree[j].boundaryscaled_split, \
				self.return_weight: return_weight, self.target_rule[0]: target_rule[0], self.target_rule[1]: target_rule[1], self.target_rule[2]: target_rule[2], self.target_rule[3]: target_rule[3], \
					self.policy_indicator: policy_indicator, self.rule_indicator: rule_indicator, self.split_indicator: split_indicator , self.target_primitive: target_primitive})

# May not need to rewrite construct_parse_tree?
	def construct_parse_tree(self,image_index):
		# WHILE WE TERMINATE THAT PARSE:

		self.painted_image = -npy.ones((self.image_size,self.image_size))
		self.alternate_painted_image = -npy.ones((self.image_size,self.image_size))
		self.alternate_predicted_labels = npy.zeros((self.image_size,self.image_size))
		
		while ((self.predicted_labels[image_index]==0).any() or (self.current_parsing_index<=len(self.parse_tree)-1)):
	
			# Forward pass of the rule policy- basically picking which rule.
			self.state = self.parse_tree[self.current_parsing_index]
			# Pick up correct portion of image.
			boundary_width = 0
			lowerx = max(0,self.state.x-boundary_width)
			upperx = min(self.image_size,self.state.x+self.state.w+boundary_width)
			lowery = max(0,self.state.y-boundary_width)
			uppery = min(self.image_size,self.state.y+self.state.h+boundary_width)

			self.image_input = self.images[image_index, lowerx:upperx, lowery:uppery, :]

			self.imagex = upperx-lowerx
			self.imagey = uppery-lowery
			self.ux = upperx
			self.uy = uppery
			self.lx = lowerx
			self.ly = lowery

			# self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))
			self.resized_image = copy.deepcopy(self.image_input)

			print("Still parsing.",len(self.parse_tree))
			# If the current non-terminal is a shape.
			if (self.state.label==0):
				self.parse_nonterminal(image_index)

			# If the current non-terminal is a region assigned a particular primitive.
			if (self.state.label==1) or (self.state.label==2):
				self.parse_primitive_terminal(image_index)
			
			self.update_plot_data(image_index)
			# self.fig.savefig("Image_{0}_Step_{1}.png".format(image_index,self.current_parsing_index),format='png',bbox_inches='tight')

	def attention_plots(self):

		self.mask = -npy.ones((self.image_size,self.image_size))
		self.display_discount = 0.8
		self.backward_discount = 0.98

		for j in range(self.current_parsing_index):
			self.dummy_state = self.parse_tree[j]
			self.mask[self.dummy_state.x:self.dummy_state.x+self.dummy_state.w,self.dummy_state.y:self.dummy_state.y+self.dummy_state.h] = -(self.backward_discount**j)
		
		for j in range(self.current_parsing_index,len(self.parse_tree)):
			self.dummy_state = self.parse_tree[j]
			self.mask[self.dummy_state.x:self.dummy_state.x+self.dummy_state.w,self.dummy_state.y:self.dummy_state.y+self.dummy_state.h] = (self.display_discount**(j-self.current_parsing_index))

	def update_plot_data(self, image_index):
	
		# if (self.predicted_labels[image_index]==1).any():
		# self.alternate_painted_image[npy.where(self.predicted_labels[image_index]==1)]=1.
		self.alternate_painted_image[npy.where(self.painted_images[image_index]==1)]=1.
		self.alternate_predicted_labels[npy.where(self.predicted_labels[image_index]==1)]=1.
		self.alternate_predicted_labels[npy.where(self.predicted_labels[image_index]==2)]=-1.

		if self.plot:
			self.fig.suptitle("Processing Image: {0}".format(image_index)) 
			
			self.attention_plots()
			self.sc0.set_data(self.mask)            
			self.sc1.set_data(self.original_images[image_index])
			self.sc2.set_data(self.true_labels[image_index])    
			# self.sc3.set_data(self.alternate_painted_image)
			self.sc3.set_data(self.alternate_predicted_labels)
			self.sc4.set_data(self.original_images[image_index])
			
			# Plotting split line segments from the parse tree.
			split_segs = []
			for j in range(len(self.parse_tree)):

				colors = ['r']

				if self.parse_tree[j].label==0:

					rule_app_map = self.remap_rule_indices(self.parse_tree[j].rule_applied)

					if (self.parse_tree[j].alter_rule_applied==1) or (self.parse_tree[j].alter_rule_applied==3):
					# if (rule_app_map==1) or (rule_app_map==3):
						sc = self.parse_tree[j].boundaryscaled_split
						split_segs.append([[self.parse_tree[j].y,self.parse_tree[j].x+sc],[self.parse_tree[j].y+self.parse_tree[j].h,self.parse_tree[j].x+sc]])
						
					if (self.parse_tree[j].alter_rule_applied==0) or (self.parse_tree[j].alter_rule_applied==2):                    
					# if (rule_app_map==0) or (rule_app_map==2):
						sc = self.parse_tree[j].boundaryscaled_split
						split_segs.append([[self.parse_tree[j].y+sc,self.parse_tree[j].x],[self.parse_tree[j].y+sc,self.parse_tree[j].x+self.parse_tree[j].w]])

				# print(split_segs) 
			split_lines0 = LineCollection(split_segs, colors='k', linewidths=2)
			split_lines1 = LineCollection(split_segs, colors='k', linewidths=2)
			split_lines2 = LineCollection(split_segs, colors='k',linewidths=2)
			split_lines3 = LineCollection(split_segs, colors='k',linewidths=2)
			
			self.split_lines0 = self.ax[0].add_collection(split_lines0)             
			self.split_lines1 = self.ax[1].add_collection(split_lines1)         
			self.split_lines2 = self.ax[2].add_collection(split_lines2)
			self.split_lines3 = self.ax[3].add_collection(split_lines3)

			if len(self.start_list)>0 and len(self.goal_list)>0:
				segs = [[npy.array([0,0]),self.start_list[0]]]
				color_index = ['k']
				linewidths = [1]

				for i in range(len(self.goal_list)-1):
					segs.append([self.start_list[i],self.goal_list[i]])
					# Paint
					color_index.append('y')
					linewidths.append(5)
					segs.append([self.goal_list[i],self.start_list[i+1]])
					# Don't paint.
					color_index.append('k')
					linewidths.append(1)
				# Add final segment.
				segs.append([self.start_list[-1],self.goal_list[-1]])
				color_index.append('y')
				linewidths.append(5)

				lines = LineCollection(segs, colors=color_index,linewidths=linewidths)
				self.lines = self.ax[4].add_collection(lines)
			
			self.fig.canvas.draw()
			# raw_input("Press any key to continue.")
			plt.pause(0.1)  

			del self.ax[0].collections[-1]
			del self.ax[1].collections[-1]          
			del self.ax[2].collections[-1]
			del self.ax[3].collections[-1]

			if len(self.ax[4].collections):
				del self.ax[4].collections[-1]	

	def define_plots(self):
		image_index = 0
		
		if self.plot:

			self.fig, self.ax = plt.subplots(1,5,sharey=True)
			self.fig.show()
			
			self.sc0 = self.ax[0].imshow(self.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			self.sc0.set_clim([-1,1])
			self.ax[0].set_title("Parse Tree")
			self.ax[0].set_adjustable('box-forced')

			self.sc1 = self.ax[1].imshow(self.original_images[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			# self.sc1.set_clim([-1,1])
			self.ax[1].set_title("Actual Image")
			self.ax[1].set_adjustable('box-forced')

			self.sc2 = self.ax[2].imshow(self.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			self.sc2.set_clim([-1,1])
			self.ax[2].set_title("True Labels")
			self.ax[2].set_adjustable('box-forced')

			self.sc3 = self.ax[3].imshow(self.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			self.sc3.set_clim([-1,1])
			self.ax[3].set_title("Predicted labels")
			self.ax[3].set_adjustable('box-forced')         

			self.sc4 = self.ax[4].imshow(self.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			self.sc4.set_clim([-1,1])
			self.ax[4].set_title("Segmented Painted Image")
			self.ax[4].set_adjustable('box-forced')         

			self.fig.canvas.draw()
			plt.pause(0.1)  

	def meta_training(self,train=True):
		
		image_index = 0
		self.painted_image = -npy.ones((self.image_size,self.image_size))
		self.predicted_labels = npy.zeros((self.num_images,self.image_size, self.image_size))
		self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))
		# self.minimum_width = self.paintwidth
		
		if self.plot:
			print(self.plot)
			self.define_plots()

		self.to_train = train
		# For all epochs
		if not(train):
			self.num_epochs=1

		# For all epochs
		for e in range(self.num_epochs):

			image_list = npy.array(range(self.num_images))
			npy.random.shuffle(image_list)            

			for jx in range(self.num_images):

				# Image index to process.
				i = image_list[jx]
				self.vertical_grad = self.gradients[i,0]
				self.horizontal_grad = self.gradients[i,1]

				self.initialize_tree()
				self.construct_parse_tree(i)
				self.propagate_rewards()                
				print("#___________________________________________________________________________")
				print("Epoch:",e,"Training Image:",jx,"TOTAL REWARD:",self.parse_tree[0].reward)

				if train:
					self.backprop(i)
				self.start_list = []
				self.goal_list = []
				
				if ((i%self.save_every)==0):
					self.save_model(e,jx/self.save_every)

			if train:
				npy.save("parsed_{0}.npy".format(e),self.predicted_labels)
				npy.save("painted_images_{0}.npy".format(e),self.painted_images)
				self.save_model(e)
			else: 
				npy.save("validation_{0}.npy".format(self.suffix),self.predicted_labels)
				npy.save("validation_painted_{0}.npy".format(self.suffix),self.painted_images)
				
			self.predicted_labels = npy.zeros((self.num_images,self.image_size,self.image_size))
			self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))

# This mapping may need to change.
	def map_rules_to_indices(self, rule_index):
		if (rule_index<=3):
			return [0,0]
		if (rule_index==4):
			return 1
		if (rule_index==5):
			return 2    

	def preprocess_images_labels(self):

	#   noise = 0.2*npy.random.rand(self.num_images,self.image_size,self.image_size)
	#   self.images[npy.where(self.images==2)]=-1
	#   self.true_labels[npy.where(self.true_labels==2)]=-1
	#   self.images += noise  

		# INSTEAD OF ADDING NOISE to the images, now we are going to normalize the images to -1 to 1 (float values).
		# Convert labels to -1 and 1 too.
		# self.true_labels = self.true_labels.astype(float)
		# self.true_labels /= self.true_labels.max()
		# self.true_labels -= 0.5
		# self.true_labels *= 2

		self.images = self.images.astype(float)
		# self.image_means = self.images.mean(axis=(0,1,2))
		# self.images -= self.image_means
		# self.images_maxes = self.images.max(axis=(0,1,2))
		# self.images /= self.images_maxes


def _activation_summary(x):
	"""Helper to create summaries for activations.

	Creates a summary that provides a histogram of activations.
	Creates a summary that measure the sparsity of activations.

	Args:
	  x: Tensor
	Returns:
	  nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = x.op.name
	# tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def parse_arguments():

	parser = argparse.ArgumentParser(description='Primitive-Aware Segmentation Argument Parsing')
	parser.add_argument('--images',dest='images',type=str)
	parser.add_argument('--labels',dest='labels',type=str)
	parser.add_argument('--size',dest='size',type=int)
	parser.add_argument('--paintwidth',dest='paintwidth',type=int)
	parser.add_argument('--minwidth',dest='minwidth',type=int)
	parser.add_argument('--lambda',dest='inter_lambda',type=float)
	parser.add_argument('--model',dest='model',type=str)
	parser.add_argument('--suffix',dest='suffix',type=str)
	parser.add_argument('--gpu',dest='gpu')
	parser.add_argument('--plot',dest='plot',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--gradient',dest='gradients',type=str)

	return parser.parse_args()

def main(args):

	args = parse_arguments()
	print(args)

	# # Create a TensorFlow session with limits on GPU usage.
	# gpu_ops = tf.GPUOptions(visible_device_list=args.gpu)
	gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list=args.gpu)
	config = tf.ConfigProto(gpu_options=gpu_ops, allow_soft_placement = True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.90
	sess = tf.Session(config=config)

	hierarchical_model = hierarchical()
	# hierarchical_model.build(sess)
	hierarchical_model.images = npy.load(args.images)
	hierarchical_model.original_images = npy.load(args.images)
	hierarchical_model.true_labels = npy.load(args.labels)
	hierarchical_model.gradients = npy.load(args.gradients)
	hierarchical_model.image_size = args.size 
	hierarchical_model.preprocess_images_labels()

	hierarchical_model.paintwidth = args.paintwidth
	hierarchical_model.minimum_width = args.minwidth
	hierarchical_model.intermittent_lambda = args.inter_lambda

	hierarchical_model.plot = args.plot
	hierarchical_model.to_train = args.train
	
	# if hierarchical_model.to_train:
	hierarchical_model.suffix = args.suffix

	if args.model:
		# hierarchical_model.initialize_tensorflow_model(sess,args.model)
		hierarchical_model.build(sess,args.model)
	else:
		# hierarchical_model.initialize_tensorflow_model(sess)
		hierarchical_model.build(sess)

	hierarchical_model.meta_training(train=args.train)

if __name__ == '__main__':
	main(sys.argv)





