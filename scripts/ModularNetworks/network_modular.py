#!/usr/bin/env python
from headers import *
from state_class import *

class hierarchical():

	# def __init__(self):

	# 	self.num_epochs = 20
	# 	self.num_images = 20000
	# 	self.current_parsing_index = 0
	# 	self.parse_tree = [parse_tree_node()]
	# 	self.paintwidth = 2
	# 	self.minimum_width = self.paintwidth
	# 	self.images = []
	# 	self.true_labels = []
	# 	self.image_size = 20
	# 	self.predicted_labels = npy.zeros((self.num_images,self.image_size, self.image_size))
    
    def __init__(self):

        self.num_epochs = 20
        self.num_images = 20000
        self.current_parsing_index = 0
        self.parse_tree = [parse_tree_node()]
        self.paintwidth = -1
        self.minimum_width = -1
        self.images = []
        self.true_labels = []
        self.image_size = -1
        self.intermittent_lambda = 0.

	def initialize_tensorflow_model(self, sess, model_file=None):

		# Define a tensorflow session.
		self.sess = sess

		# Number of layers. 
		self.num_layers = 5
		self.num_fc_layers = 2
		self.conv_sizes = 3*npy.ones((self.num_layers),dtype=int)		
		self.conv_num_filters = npy.array([1,20,20,20,20,20],dtype=int)

		# Placeholders
		self.input = tf.placeholder(tf.float32,shape=[1,self.image_size,self.image_size,1],name='input')
		
		############ CONV LAYERS #######################################################################

		# Declare common conv variables.
		self.W_conv = [[] for i in range(self.num_layers)]
		self.b_conv = [[] for i in range(self.num_layers)]
		# Defining conv layers.
		self.conv = [[] for i in range(self.num_layers)]
		self.relu_conv = [[] for i in range(self.num_layers)]

		# Defining variables. 
		for i in range(self.num_layers):
			self.W_conv[i] = tf.Variable(tf.truncated_normal([self.conv_sizes[i],self.conv_sizes[i], self.conv_num_filters[i], self.conv_num_filters[i+1]],stddev=0.1),name='W_conv{0}'.format(i+1))
			self.b_conv[i] = tf.Variable(tf.constant(0.1,shape=[self.conv_num_filters[i]]),name='b_conv{0}'.format(i+1))

		# Defining first conv layer.
		self.conv[0] = tf.add(tf.nn.conv2d(self.input,self.W_conv[0],strides=[1,1,1,1],padding='VALID'),self.b_conv[0],name='conv1')
		self.relu_conv[0] = tf.nn.relu(self.conv[0],name='relu0')
		
		# Defining subsequent conv layers.
		for i in range(1,self.num_layers):
			self.conv[i] = tf.add(tf.nn.conv2d(self.conv[i-1],self.W_conv[i],strides=[1,1,1,1],padding='VALID'),self.b_conv[i],name='conv{0}'.format(i+1))
			self.relu_conv[i] = tf.nn.relu(self.conv[i],name='relu{0}'.format(i+1))
		
		################################################################################################

		########## COMMON FC LAYERS ####################################################################

		self.fc_input_shape = 5*5*self.conv_num_filters[-1]
		
		# Rule stream
		self.rule_num_fclayers = 2
		self.rule_num_hidden = 80
		self.rule_num_branches = 4
		self.rule_fc_shapes = [[self.fc_input_shape,self.rule_num_hidden,6],[self.fc_input_shape,self.rule_num_hidden,4],[self.fc_input_shape,self.rule_num_hidden,4],[self.fc_input_shape,self.rule_num_hidden,2]]

		# Split stream
		self.split_num_fclayers = 2
		self.split_num_hidden = 50
		self.split_num_branches = 2
		self.split_fc_shapes = [self.fc_input_shape,self.split_num_hidden,2]

		# Primitive stream
		self.primitive_num_fclayers = 2
		self.primitive_num_hidden = 50
		self.number_primitives = 4
		self.primitive_fc_shapes = [self.fc_input_shape,self.primitive_num_hidden,self.number_primitives]

		# Reshape FC input.
		self.fc_input = tf.reshape(self.relu_conv[-1],[-1,self.fc_input_shape],name='fc_input')
		
		################################################################################################

		########## RULE FC LAYERS ######################################################################
		
		# Defining FC layer variables lists.
		self.W_rule_fc = [[[] for i in range(self.rule_num_fclayers)] for j in range(self.rule_num_branches)]
		self.b_rule_fc = [[[] for i in range(self.rule_num_fclayers)] for j in range(self.rule_num_branches)]

		# Defining rule_fc layers.
		self.rule_fc = [[] for i in range(self.rule_num_fclayers)] for j in range(self.rule_num_branches)]
		self.rule_probabilities = [[] for j in range(self.rule_num_branches)]
		self.rule_dist = [[] for j in range(self.rule_num_branches)]

		# Can maintain a single sample_rule and sampled_rule for all of the branches, because we will use tf.case for each.
		self.rule_indicator = tf.placeholder(tf.int32,name='rule_indicator')
		self.sampled_rule = tf.placeholder(tf.int32,name='sampled_rule')

		# Defining Rule FC variables.
		for j in range(self.rule_num_branches):
			for i in range(self.num_fc_layers):
				self.W_rule_fc[j][i] = tf.Variable(tf.truncated_normal([self.rule_fc_shapes[j][i],self.rule_fc_shapes[j][i+1]],stddev=0.1),name='W_rulefc_branch{0}_layer{1}'.format(j,i+1))
				self.b_rule_fc[j][i] = tf.Variable(tf.constant(0.1,shape=[self.rule_fc_shapes[j][i+1]]),name='b_rulefc_branch{0}_layer{1}'.format(j,i+1))
			# self.sampled_rule[j] = tf.placeholder(tf.int32)

		# Defining Rule FC layers.
		for j in range(self.rule_num_branches):
			self.rule_fc[j][0] = tf.nn.relu(tf.add(tf.matmul(self.fc_input,self.W_rule_fc[j][0]),self.b_rule_fc[j][0]),name='rule_fc_branch{0}_layer0'.format(j))
			self.rule_fc[j][1] = tf.add(tf.matmul(self.rule_fc[j][0],self.W_rule_fc[j][1]),self.b_rule_fc[j][1],name='rule_fc_branch{0}_layer1'.format(j))
			self.rule_probabilities[j] = tf.nn.softmax(self.rule_fc[j][1],name='rule_probabilities_branch{0}'.format(j))
			self.rule_dist[j] = tf.contrib.distributions.Categorical(probs=self.rule_probabilities[j],name='rule_dist_branch{0}'.format(j))
			# self.sample_rule[j] = self.rule_dist[j].sample()
		
		# This is the heart of the routing. We select which set of parameters we sample the rule from.
		# Default needs to be a lambda function because we're providing arguments to it. 
		self.sample_rule = tf.case({tf.equal(self.rule_indicator,0): self.rule_dist[0].sample, tf.equal(self.rule_indicator,1): self.rule_dist[1].sample, 
									tf.equal(self.rule_indicator,2): self.rule_dist[2].sample, tf.equal(self.rule_indicator,3): self.rule_dist[3].sample},default=lambda: -tf.ones(1),exclusive=True,name='sample_rule')

		################################################################################################

		########### SPLIT FC LAYERS ####################################################################

		# Defining FC layer variables lists.
		self.W_split_fc = [[[] for i in range(self.split_num_fclayers)] for j in range(self.split_num_branches)]
		self.b_split_fc = [[[] for i in range(self.split_num_fclayers)] for j in range(self.split_num_branches)]

		# Defining split_fc layers.
		self.split_fc = [[] for i in range(self.split_num_fclayers)] for j in range(self.split_num_branches)]
		self.split_mean = [[] for j in range(self.split_num_branches)]
		self.split_cov = [[] for j in range(self.split_num_branches)]
		self.split_dist = [[] for j in range(self.split_num_branches)]

		# Similarly to rules, we can use one sample_split, because we use tf.case.
		self.split_indicator = tf.placeholder(tf.int32,name='split_indicator')
		self.sampled_split = tf.placeholder(tf.float32,name='sampled_split')

		# Defining split FC variables.
		for j in range(self.split_num_branches):
			for i in range(self.split_num_fclayers):
				self.W_split_fc[j][i] = tf.Variable(tf.truncated_normal([self.split_fc_shapes[j][i],self.split_fc_shapes[j][i+1]],stddev=0.1),name='W_splitfc_branch{0}_layer{1}'.format(j,i+1))
				self.b_split_fc[j][i] = tf.Variable(tf.constant(0.1,shape=[self.split_fc_shapes[j][i+1]]),name='b_splitfc_branch{0}_layer{1}'.format(j,i+1))

		# Defining split FC layers.
		for j in range(self.split_num_branches):
			self.split_fc[j][0] = tf.nn.relu(tf.add(tf.matmul(self.fc_input,self.W_split_fc[j][0]),self.b_split_fc[j][0]),name='split_fc_branch{0}_layer0'.format(j))
			self.split_fc[j][1] = tf.add(tf.matmul(self.split_fc[j][0],self.W_split_fc[j][1]),self.b_split_fc[j][1],name='split_fc_branch{0}_layer1'.format(j))

			self.split_mean[j] = tf.nn.sigmoid(self.split_fc[j][1][0,0])
			# If the variance is learnt.
			self.split_cov[j] = tf.nn.sigmoid(self.split_fc[j][1][0,1])
			# Defining distributions for each.
			self.split_dist[j] = tf.contrib.distributions.Normal(loc=self.split_mean[j],scale=self.split_cov[j],name='split_dist_branch{0}'.format(j))
			# self.sample_split[j] = self.split_dist[j].sample()

		# This is the heart of the routing. We select which set of parameters we sample the split location from. 
		# Default needs to be a lambda function because we're providing arguments to it. 
		self.sample_split = tf.case({tf.equal(self.split_indicator,0): self.split_dist[0].sample,tf.equal(self.split_indicator,1): self.split_dist[1].sample},default=lambda: -tf.ones(1),exclusive=True,name='sample_split')

		################################################################################################

		########## PRIMITIVE FC LAYERS #################################################################

		# Defining FC layaer for primitive stream.
		self.W_primitive_fc = [[] for i in range(self.primitive_num_fclayers)]
		self.b_primitive_fc = [[] for i in range(self.primitive_num_fclayers)]

		# Defining primitive fc layers.
		self.primitive_fc = [[] for i in range(self.primitive_num_fclayers)]

		# Defining variables:
		for i in range(self.primitive_num_fclayers):
			self.W_primitive_fc[j][i] = tf.Variable(tf.truncated_normal([self.primitive_fc_shapes[j][i],self.primitive_fc_shapes[j][i+1]],stddev=0.1),name='W_primitivefc_branch{0}_layer{1}'.format(j,i+1))
			self.b_primitive_fc[j][i] = tf.Variable(tf.constant(0.1,shape=[self.primitive_fc_shapes[j][i+1]]),name='b_primitivefc_branch{0}_layer{1}'.format(j,i+1))
		
		# Defining primitive FC layers.
		self.primitive_fc[0] = tf.nn.relu(tf.add(tf.matmul(self.fc_input,self.W_primitive_fc[0]),self.b_primitive_fc[0]),name='primitve_fc_layer0')
		self.primitive_fc[1] = tf.add(tf.matmul(self.primitive_fc[0],self.W_primitive_fc[1]),self.b_primitive_fc[1]),name='primitve_fc_layer1')
		self.primitive_probabilities = tf.nn.softmax(self.primitive_fc[-1],name='primitive_probabilities')
		
		# Defining categorical distribution for primitives.
		self.primitive_dist = tf.contrib.distributions.Categorical(probs=self.primitive_probabilities,name='primitive_distribution')
		
		################################################################################################

		########### NOW MOVING TO THE LOSS FUNCTIONS ###################################################

		self.return_weight = tf.placeholder(tf.float32,name='return_weight')

		######## For rule stream:##########
		self.target_rule = [[] for j in range(self.rule_num_branches)] 
		self.rule_loss_branch = [[] for j in range(self.rule_num_branches)]

		# Defining a log probability loss for each of the rule policy branches.
		for j in range(self.rule_num_branches):
			self.target_rule[j] = tf.placeholder(tf.float32,shape=(self.rule_fc_shapes[j][-1]))
			# self.rule_loss_branch[j] = tf.multiply(self.return_weight,tf.nn.softmax_cross_entropy_with_logits(labels=self.target_rule[j],logits=self.rule_fc[j][-1]),name='rule_loss_branch{0}'.format(j))			
			self.rule_loss_branch[j] = tf.nn.softmax_cross_entropy_with_logits(labels=self.target_rule[j],logits=self.rule_fc[j][-1],name='rule_loss_branch{0}'.format(j))

		# Defining a loss that selects which branch to back-propagate into.
		self.rule_loss = tf.case({tf.equal(self.rule_indicator,0): self.rule_loss_branch[0], tf.equal(self.rule_indicator,1): self.rule_loss_branch[1], 
									tf.equal(self.rule_indicator,2): self.rule_loss_branch[2], tf.equal(self.rule_indicator,3): self.rule_loss_branch[3]},default=lambda: -tf.ones(1),exclusive=True,name='rule_loss')

		######## For split stream:#########
		self.split_loss_branch = [[] for j in range(self.split_num_branches)]

		for j in range(self.split_num_branches):
			# self.split_loss_branch = -tf.multply(self.return_weight,self.split_dist[j].log_prob(self.sampled_split),name='split_loss_branch{0}'.format(j))
			self.split_loss_branch = -self.split_dist[j].log_prob(self.sampled_split)

		# Now defining a split loss that selects which branch to back-propagate into.
		self.split_loss = tf.case({tf.equal(self.split_indicator,0): self.split_dist[0].sample,tf.equal(self.split_indicator,1): self.split_dist[1].sample},default=lambda: -tf.ones(1),exclusive=True,name='sample_split')

		######### For primitive stream ####

		self.target_primitives = tf.placeholder(tf.float32,shape=(self.number_primitives))
		self.primitive_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.target_primitives,logits=self.primitive_fc[1],name='primitive_loss')

		######### COMBINED LOSS ###########

		self.policy_indicator = tf.placeholder(tf.int32)
        # There is a fixed set of policy branches that can be applied together. 
        # A split rule is applied with the split location --> Case 1
        # An assignment rule is run on its own --> Case 2
        # A primitive is run on its own --> Case 3
        
		self.selected_loss = tf.case({tf.equal(self.policy_indicator,0): self.rule_loss+self.split_loss, tf.equal(self.policy_indicator,1): self.rule_loss, tf.equal(self.policy_indicator,2): self.primitive_loss},default=lambda: tf.ones(1), exclusive=True,name='selected_loss')
        self.total_loss = tf.multply(self.return_weight,self.selected_loss,name='total_loss')

		#################################################################################################

