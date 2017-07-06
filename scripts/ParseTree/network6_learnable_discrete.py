#!/usr/bin/env python
from headers import *

# Define a class for the parse tree / rule / etc? 
class parse_tree_node():
	# def __init__(self, label=-1, x=-1, y=-1,w=-1,h=-1,backward_index=-1,rule_applied=-1, split=-1, start=npy.array([-1,-1]), goal=npy.array([-1,-1])):
	def __init__(self, label=-1, x=-1, y=-1,w=-1,h=-1,backward_index=-1,rule_applied=-1, split=0, start=npy.array([-1,-1]), goal=npy.array([-1,-1])):
		self.label = label
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.backward_index = backward_index
		self.rule_applied = rule_applied
		self.split = split
		self.reward = 0.
		self.gradient_values = npy.ones((1,20))/20

	def disp(self):
		print("Label:", self.label)
		print("X:",self.x,"Y:",self.y,"W:",self.w,"H:",self.h)
		print("Backward Index:",self.backward_index)
		print("Reward:",self.reward)
		print("Rule:",self.rule_applied,"Split:",self.split)
		print("____________________________________________")

class hierarchical():

	def __init__(self):

		self.num_epochs = 20
		self.num_images = 20000
		self.current_parsing_index = 0
		self.parse_tree = [parse_tree_node()]
		self.paintwidth=2
		self.images = []
		self.true_labels = []
		self.image_size = 20
		self.predicted_labels = npy.zeros((self.num_images,self.image_size, self.image_size))

	def initialize_tensorflow_model(self, sess, model_file=None):

		# Initializing the session.
		self.sess = sess

		# Image size and other architectural parameters. 
		self.conv1_size = 3	
		self.conv1_num_filters = 20
		self.conv2_size = 3	
		self.conv2_num_filters = 20
		self.conv3_size = 3	
		self.conv3_num_filters = 20
		self.conv4_size = 3	
		self.conv4_num_filters = 20
		self.conv5_size = 3	
		self.conv5_num_filters = 20

		# Placeholders
		self.input = tf.placeholder(tf.float32,shape=[1,self.image_size,self.image_size,1],name='input')

		# Convolutional layers: 
		# Layer 1
		self.W_conv1 = tf.Variable(tf.truncated_normal([self.conv1_size,self.conv1_size, 1, self.conv1_num_filters],stddev=0.1),name='W_conv1')
		self.b_conv1 = tf.Variable(tf.constant(0.1,shape=[self.conv1_num_filters]),name='b_conv1')
		self.conv1 = tf.add(tf.nn.conv2d(self.input,self.W_conv1,strides=[1,1,1,1],padding='VALID'),self.b_conv1,name='conv1')
		self.relu_conv1 = tf.nn.relu(self.conv1)

		# Layer 2 
		self.W_conv2 = tf.Variable(tf.truncated_normal([self.conv2_size,self.conv2_size,self.conv1_num_filters,self.conv2_num_filters],stddev=0.1),name='W_conv2')
		self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[self.conv2_num_filters]),name='b_conv2')
		self.conv2 = tf.add(tf.nn.conv2d(self.relu_conv1,self.W_conv2,strides=[1,1,1,1],padding='VALID'),self.b_conv2,name='conv2')
		self.relu_conv2 = tf.nn.relu(self.conv2)

		# Layer 3
		self.W_conv3 = tf.Variable(tf.truncated_normal([self.conv3_size,self.conv3_size,self.conv2_num_filters,self.conv3_num_filters],stddev=0.1),name='W_conv3')
		self.b_conv3 = tf.Variable(tf.constant(0.1,shape=[self.conv3_num_filters]),name='b_conv3')
		self.conv3 = tf.add(tf.nn.conv2d(self.relu_conv2,self.W_conv3,strides=[1,1,1,1],padding='VALID'),self.b_conv3,name='conv3')
		self.relu_conv3 = tf.nn.relu(self.conv3)

		# Layer 4
		self.W_conv4 = tf.Variable(tf.truncated_normal([self.conv4_size,self.conv4_size,self.conv3_num_filters,self.conv4_num_filters],stddev=0.1),name='W_conv4')
		self.b_conv4 = tf.Variable(tf.constant(0.1,shape=[self.conv4_num_filters]),name='b_conv4')
		self.conv4 = tf.add(tf.nn.conv2d(self.relu_conv3,self.W_conv4,strides=[1,1,1,1],padding='VALID'),self.b_conv4,name='conv4')
		self.relu_conv4 = tf.nn.relu(self.conv4)

		# Layer 5
		self.W_conv5 = tf.Variable(tf.truncated_normal([self.conv5_size,self.conv5_size,self.conv4_num_filters,self.conv5_num_filters],stddev=0.1),name='W_conv5')
		self.b_conv5 = tf.Variable(tf.constant(0.1,shape=[self.conv5_num_filters]),name='b_conv5')
		# self.conv5 = tf.add(tf.nn.conv2d(self.relu_conv4,self.W_conv5,strides=[1,2,2,1],padding='VALID'),self.b_conv5,name='conv5')
		self.conv5 = tf.add(tf.nn.conv2d(self.relu_conv4,self.W_conv5,strides=[1,1,1,1],padding='VALID'),self.b_conv5,name='conv5')
		self.relu_conv5 = tf.nn.relu(self.conv5)

		# Now going to flatten this and move to a fully connected layer.s
		self.fc_input_shape = self.relu_conv5.shape[1]
		self.fc_input_shape = 10*10*self.conv5_num_filters
		# self.fc_input_shape = 5*5*self.conv5_num_filters
		self.relu_conv5_flat = tf.reshape(self.relu_conv5,[-1,self.fc_input_shape])

		# Going to split into 4 streams: RULE, SPLIT, START and GOAL
		self.fcs1_l1_shape = 120
		self.W_fcs1_l1 = tf.Variable(tf.truncated_normal([self.fc_input_shape,self.fcs1_l1_shape],stddev=0.1),name='W_fcs1_l1')
		self.b_fcs1_l1 = tf.Variable(tf.constant(0.1,shape=[self.fcs1_l1_shape]),name='b_fcs1_l1')
		self.fcs1_l1 = tf.nn.relu(tf.add(tf.matmul(self.relu_conv5_flat,self.W_fcs1_l1),self.b_fcs1_l1),name='fcs1_l1')

		# 2nd FC layer: RULE Output:
		self.number_primitives = 1
		# Now we have shifted to the 4 rule version of this: 
		# Horizontal split rule into two shapes / # Vertical split rule into two shapes / # Assignment rule to region with primitive /	# Assignment rule to region without primitive.

		self.fcs1_output_shape = 1*self.number_primitives+5
		self.W_fcs1_l2 = tf.Variable(tf.truncated_normal([self.fcs1_l1_shape,self.fcs1_output_shape],stddev=0.1),name='W_fcs1_l2')
		self.b_fcs1_l2 = tf.Variable(tf.constant(0.1,shape=[self.fcs1_output_shape]),name='b_fcs1_l2')
		self.fcs1_presoftmax = tf.add(tf.matmul(self.fcs1_l1,self.W_fcs1_l2),self.b_fcs1_l2,name='fcs1_presoftmax')
		self.rule_probabilities = tf.nn.softmax(self.fcs1_presoftmax,name='softmax')
		
		# CREATING GRADIENT STREAM: CATEGORICAL PROBABILITIES:
		self.gradient_values = tf.placeholder(tf.float32,shape=(None,self.image_size),name='gradient_values')

		# First hidden layer:
		self.hidden_fc1 = 40		
		self.W_fc1 = tf.Variable(tf.truncated_normal([self.image_size,self.hidden_fc1],stddev=0.1),name='W_fc1')
		self.b_fc1 = tf.Variable(tf.constant(0.1,shape=[self.hidden_fc1]),name='b_fc1')
		self.relu_fc1 = tf.nn.relu(tf.add(tf.matmul(self.gradient_values,self.W_fc1),self.b_fc1),name='gradient_fc1')

		# Second hidden layer:
		self.hidden_fc2 = 70
		self.W_fc2 = tf.Variable(tf.truncated_normal([self.hidden_fc1,self.hidden_fc2],stddev=0.1),name='W_fc2')
		self.b_fc2 = tf.Variable(tf.constant(0.1,shape=[self.hidden_fc2]),name='b_fc2')
		self.relu_fc2 = tf.nn.relu(tf.add(tf.matmul(self.relu_fc1,self.W_fc2),self.b_fc2),name='gradient_fc2')

		self.W_fc3 = tf.Variable(tf.truncated_normal([self.hidden_fc2,self.image_size],stddev=0.1),name='W_fc3')
		self.b_fc3 = tf.Variable(tf.constant(0.1,shape=[self.image_size]),name='b_fc3')
		self.categorical_probabilities = tf.nn.softmax(tf.add(tf.matmul(self.relu_fc2,self.W_fc3),self.b_fc3),name='gradient_fc3')		

		# Vector of probabilities along ONE dimension.
		# self.categorical_probabilities = tf.placeholder(tf.float32,shape=(None,self.image_size),name='categorical_probabilities')
		self.split_dist = tf.contrib.distributions.Categorical(probs=self.categorical_probabilities)
		
		# Sampling a goal and a split. Remember, this should still just be defining an operation, not actually sampling.
		# We evaluate this to retrieve a sample goal / split location. 
		self.sample_split = self.split_dist.sample()
		# Also maintaining placeholders for scaling, converting to integer, and back to float.
		self.sampled_split = tf.placeholder(tf.int32,shape=(None),name='sampled_split')

		# Defining training ops. 
		self.rule_return_weight = tf.placeholder(tf.float32,shape=(None),name='rule_return_weight')
		self.split_return_weight = tf.placeholder(tf.float32,shape=(None),name='split_return_weight')
		self.target_rule = tf.placeholder(tf.float32,shape=( self.fcs1_output_shape),name='target_rule')

		# Defining the loss for each of the 3 streams, rule, split and goal.
		# Rule loss is the negative cross entropy between the rule probabilities and the chosen rule as a one-hot encoded vector. 
		# Weighted by the return obtained. This is just the negative log probability of the selected action.

		# NO NEGATIVE SIGN HERE - 13/6
		self.rule_loss = tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=self.target_rule,logits=self.fcs1_presoftmax),self.rule_return_weight, name='rule_loss')

		# The split loss is the negative log probability of the chosen split, weighted by the return obtained.
		# TRYING SPLIT LOSS WITH NEGATIVE SIGN 30/06
		self.split_loss = -tf.multiply(self.split_dist.log_prob(self.sampled_split),self.split_return_weight,name='split_loss')
		# The total loss is the sum of individual losses.
		self.split_loss_weightage = 0.1
		self.total_loss = tf.add(self.rule_loss,self.split_loss_weightage*self.split_loss,name='total_loss')

		# Creating summaries to log the losses.
		self.rule_loss_summary = tf.summary.scalar('Rule_Loss',self.rule_loss[0])
		self.split_loss_summary = tf.summary.scalar('Split_Loss',self.split_loss[0])
		self.total_loss_summary = tf.summary.scalar('Total_Loss',self.total_loss[0])

		self.merge_summaries = tf.summary.merge_all()
		# Creating a training operation to minimize the total loss.
		self.train = tf.train.AdamOptimizer(1e-4).minimize(self.total_loss,name='Adam_Optimizer')
		# self.train = tf.train.AdamOptimizer(1e-4).minimize(self.rule_loss,name='Adam_Optimizer')

		# Writing graph and other summaries in tensorflow.
		self.writer = tf.summary.FileWriter('training',self.sess.graph)
		# Creating a saver object to save models.
		self.saver = tf.train.Saver(max_to_keep=None)

		if model_file:
			self.saver.restore(self.sess,model_file)
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)

	def save_model(self, model_index):
		save_path = self.saver.save(self.sess,'saved_models/model_{0}.ckpt'.format(model_index))

	def initialize_tree(self):
		self.current_parsing_index = 0
		self.parse_tree = [parse_tree_node()]
		self.parse_tree[self.current_parsing_index]=self.state

	def insert_node(self, state, index):
		self.parse_tree.insert(index,state)

	def parse_nonterminal(self, image_index):

		rule_probabilities = self.sess.run(self.rule_probabilities,feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
		# Should it be an epsilon-greedy policy? 

		# SAMPLING A SPLIT LOCATION
		split_location = -1

		# CHANGING THIS NOW TO BAN SPLITS FOR REGIONS SMALLER THAN: MINIMUM_WIDTH; and not just if ==1.
		self.minimum_width = 3 
		
		epislon = 1e-5
		rule_probabilities += epislon

		if (self.state.h<=self.minimum_width):
			rule_probabilities[0][[0,2]]=0.
		if (self.state.w<=self.minimum_width):
			rule_probabilities[0][[1,3]]=0.

		rule_probabilities[0]/=rule_probabilities[0].sum()
		selected_rule = npy.random.choice(range(self.fcs1_output_shape),p=rule_probabilities[0])
		indices = self.map_rules_to_indices(selected_rule)

		# If it is a split rule:
		if selected_rule<=3:
			# Apply the rule: if the rule number is even, it is a vertical split and if the current non-terminal to be parsed is taller than 1 unit:
			if ((selected_rule==0) or (selected_rule==2)):
				counter = 0

				# REMEMBER, h is along y, w is along x (transposed), # FOR THESE RULES, use y_gradient
				while (split_location<=0)or(split_location>=self.state.h):				

					categorical_prob_softmax = self.sess.run(self.categorical_probabilities, feed_dict={self.gradient_values: self.y_gradients.reshape((1,20))})[0]								
					epsilon = 0.001
					categorical_prob_softmax+=epsilon
					categorical_prob_softmax[0] = 0.
					categorical_prob_softmax[-1] = 0.
					categorical_prob_softmax /= categorical_prob_softmax.sum()
					split_location = npy.random.choice(range(20),p=categorical_prob_softmax)
					split_location = int(float(self.state.h*split_location)/20)				
					counter +=1
					if counter>=25:
						print("PREINT:",split_location,self.state.h)

					if split_location>=10:
						split_location = int(npy.floor(float(self.state.h*split_location)/20))
					else:
						split_location = int(npy.ceil(float(self.state.h*split_location)/20))
					
					if counter>=25:
						print("POSTINT:",split_location,self.state.h)
					
				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=self.state.w,h=split_location,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x,y=self.state.y+split_location,w=self.state.w,h=self.state.h-split_location,backward_index=self.current_parsing_index)

			if ((selected_rule==1) or (selected_rule==3)):
				counter = 0

				# REMEMBER, h is along y, w is along x (transposed), # FOR THESE RULES, use x_gradient
				while (split_location<=0)or(split_location>=self.state.w):				

					categorical_prob_softmax = self.sess.run(self.categorical_probabilities, feed_dict={self.gradient_values: self.x_gradients.reshape((1,20))})[0]
					epsilon = 0.001
					categorical_prob_softmax+=epsilon
					categorical_prob_softmax[0] = 0.
					categorical_prob_softmax[-1] = 0.
					categorical_prob_softmax /= categorical_prob_softmax.sum()
					split_location = npy.random.choice(range(20),p=categorical_prob_softmax)				
					counter +=1
					if counter>=25:
						print("PREINT:",split_location,self.state.w)

					if split_location>=10:
						split_location = int(npy.floor(float(self.state.w*split_location)/20))
					else:
						split_location = int(npy.ceil(float(self.state.w*split_location)/20))

					if counter>=25:
						print("POSTINT:",split_location,self.state.w)

				
				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=split_location,h=self.state.h,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x+split_location,y=self.state.y,w=self.state.w-split_location,h=self.state.h,backward_index=self.current_parsing_index)
				
			# Update current parse tree with split location and rule applied.
			self.parse_tree[self.current_parsing_index].split=split_location
			self.parse_tree[self.current_parsing_index].rule_applied=selected_rule
			self.parse_tree[self.current_parsing_index].gradient_values[0] = categorical_prob_softmax

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

		elif selected_rule>=4:
 
			# Create a parse tree node object.
			s1 = copy.deepcopy(self.parse_tree[self.current_parsing_index])
			# Change label.
			s1.label=selected_rule-3
			# Change the backward index.
			s1.backward_index = self.current_parsing_index

			# Update current parse tree with rule applied.
			self.parse_tree[self.current_parsing_index].rule_applied = selected_rule

			# Insert node into parse tree.
			self.insert_node(s1,self.current_parsing_index+1)
			self.current_parsing_index+=1						
			self.predicted_labels[image_index,s1.x:s1.x+s1.w,s1.y:s1.y+s1.h] = s1.label

	def parse_primitive_terminal(self):
		# Sample a goal location.
		self.current_parsing_index+=1

	def propagate_rewards(self):
		# Traverse the tree in reverse order, accumulate rewards into parent nodes recursively as sum of rewards of children.
		# This is actually the return accumulated by any particular decision.
		# Now we are discounting based on the depth of the tree (not just sequence in episode)
		self.gamma = 0.98
		for j in reversed(range(len(self.parse_tree))):	
			if (self.parse_tree[j].backward_index>=0):
				self.parse_tree[self.parse_tree[j].backward_index].reward += self.parse_tree[j].reward*self.gamma

		for j in range(len(self.parse_tree)):
			self.parse_tree[j].reward /= (self.parse_tree[j].w*self.parse_tree[j].h)

	def terminal_reward_nostartgoal(self, image_index):

		if self.state.label==1:
			self.painted_image[self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h] = 1

		self.state.reward = (self.true_labels[image_index, self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]*self.painted_image[self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]).sum()
		# self.state.reward = -(abs(self.true_labels[image_index, self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]-self.painted_image[self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h])).sum()		

	def compute_rewards(self, image_index):
		# For all terminal symbols only.
		# Rectange intersection
		self.painted_image = -npy.ones((self.image_size,self.image_size))
	
		for j in range(len(self.parse_tree)):
			# Assign state.
			self.state = copy.deepcopy(self.parse_tree[j])

			# For every node in the tree, we know the ground truth image labels.
			# We will compute the reward as:
			# To be painted (-1 for no, 1 for yes)
			# Whether it was painted (-1 for no or 1 for yes)

			if self.parse_tree[j].label==1 or self.parse_tree[j].label==2:
				self.terminal_reward_nostartgoal(image_index)

			self.parse_tree[j].reward = copy.deepcopy(self.state.reward)

	def backprop(self, image_index, epoch):
		# Stochastic gradient descent; variable parse tree length means stochastically is better than batch. 

		# NOW CHANGING TO 4 RULE SYSTEM.
		target_rule = npy.zeros(self.fcs1_output_shape)

		for j in range(len(self.parse_tree)):

			self.state = self.parse_tree[j]
			
			boundary_width = 2
			lowerx = max(0,self.state.x-boundary_width)
			upperx = min(self.image_size,self.state.x+self.state.w+boundary_width)
			lowery = max(0,self.state.y-boundary_width)
			uppery = min(self.image_size,self.state.y+self.state.h+boundary_width)
			
			# Pick up correct portion of image.
			self.image_input = self.images[image_index, lowerx:upperx, lowery:uppery]
			self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))
			self.x_gradients, self.y_gradients = npy.gradient(self.resized_image)
			self.x_gradients = abs(self.x_gradients).sum(axis=1)
			self.y_gradients = abs(self.y_gradients).sum(axis=0)

			rule_weight = 0
			split_weight = 0
			target_rule = npy.zeros(self.fcs1_output_shape)

			# MUST PARSE EVERY NODE
			if self.parse_tree[j].label==0:
				rule_weight = self.parse_tree[j].reward
				target_rule[self.parse_tree[j].rule_applied] = 1.
				if self.parse_tree[j].rule_applied<=3:
					split_weight = self.parse_tree[j].reward
				# Here ,we only backprop for shapes, since we only choose actions for shapese.
				merged_summaries, rule_loss, _ = self.sess.run([self.merge_summaries, self.rule_loss, self.train], \
					feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1), self.rule_return_weight: rule_weight, \
					self.target_rule: target_rule, self.split_return_weight: split_weight, self.sampled_split: self.parse_tree[j].split, \
					self.gradient_values: self.parse_tree[j].gradient_values})

			# print("LOSS VALUES:",rule_loss, split_loss)
				# rule_loss, split_loss, total_loss, _ = self.sess.run([self.rule_loss, self.split_loss, self.total_loss, self.train], \
				# 	feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1), self.rule_return_weight: rule_weight, \
				# 	self.target_rule: target_rule, self.split_return_weight: split_weight, self.sampled_split: self.parse_tree[j].split})

				self.writer.add_summary(merged_summaries, self.num_images*epoch+image_index)

	def construct_parse_tree(self,image_index):
		# WHILE WE TERMINATE THAT PARSE:

		self.painted_image = -npy.ones((self.image_size,self.image_size))
		self.alternate_painted_image = -npy.ones((self.image_size,self.image_size))
		self.alternate_predicted_labels = npy.zeros((self.image_size,self.image_size))

		while ((self.predicted_labels[image_index]==0).any() or (self.current_parsing_index<=len(self.parse_tree)-1)):

			# Forward pass of the rule policy- basically picking which rule.
			self.state = self.parse_tree[self.current_parsing_index]

			# Pick up correct portion of image.
			boundary_width = 2
			lowerx = max(0,self.state.x-boundary_width)
			upperx = min(self.image_size,self.state.x+self.state.w+boundary_width)
			lowery = max(0,self.state.y-boundary_width)
			uppery = min(self.image_size,self.state.y+self.state.h+boundary_width)

			self.image_input = self.images[image_index, lowerx:upperx, lowery:uppery]
			self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))
			self.x_gradients, self.y_gradients = npy.gradient(self.resized_image)
			self.x_gradients = abs(self.x_gradients).sum(axis=1)
			self.y_gradients = abs(self.y_gradients).sum(axis=0)

			# If the current non-terminal is a shape.
			if (self.state.label==0):
				self.parse_nonterminal(image_index)

			# If the current non-terminal is a region assigned a particular primitive.
			if (self.state.label==1):
				self.parse_primitive_terminal()
			
			if (self.state.label==2):
				self.current_parsing_index+=1

			self.alternate_painted_image[npy.where(self.predicted_labels[image_index]==1)]=1.
			self.alternate_predicted_labels[npy.where(self.predicted_labels[image_index]==1)]=2.
			self.alternate_predicted_labels[npy.where(self.predicted_labels[image_index]==2)]=1.

			self.update_plot_data(image_index)

	def update_plot_data(self, image_index):
		if self.plot:
			self.fig.suptitle("Processing Image: {0}".format(image_index))
			self.sc1.set_data(self.alternate_predicted_labels)
			self.sc2.set_data(self.true_labels[image_index])
			self.sc3.set_data(self.alternate_painted_image)
			self.sc4.set_data(self.images[image_index])
			self.fig.canvas.draw()
			plt.pause(0.001)

	def define_plots(self):

		image_index = 0
		if self.plot:
			self.fig, self.ax = plt.subplots(1,4,sharey=True)
			self.fig.show()
			
			self.sc1 = self.ax[0].imshow(self.predicted_labels[image_index],aspect='equal')
			self.sc1.set_clim([0,2])
			# self.fig.colorbar(sc1, self.ax=self.ax[0])
			self.ax[0].set_title("Predicted Labels")
			self.ax[0].set_adjustable('box-forced')

			self.sc2 = self.ax[1].imshow(self.true_labels[image_index],aspect='equal')
			self.sc2.set_clim([-1,1])
			# self.fig.colorbar(sc2, self.ax=self.ax[1])
			self.ax[1].set_title("True Labels")
			self.ax[1].set_adjustable('box-forced')

			self.sc3 =self.ax[2].imshow(self.painted_image,aspect='equal')
			self.sc3.set_clim([-1,1])
			# self.fig.colorbar(sc3, self.ax=self.ax[2])
			self.ax[2].set_title("Painted Image")
			self.ax[2].set_adjustable('box-forced')

			self.sc4 = self.ax[3].imshow(self.images[image_index],aspect='equal')
			self.sc4.set_clim([-1,1])
			# self.fig.colorbar(sc4,self.ax=self.ax[3])
			self.ax[3].set_title("Actual Image")
			self.ax[3].set_adjustable('box-forced')
			# plt.draw()
			self.fig.canvas.draw()
			plt.pause(0.001)

	def meta_training(self, train=True):

		image_index = 0
		self.painted_image = -npy.ones((self.image_size,self.image_size))
		self.define_plots()

		# For all epochs
		if not(train):
			self.num_epochs=1
	
		for e in range(self.num_epochs):	
			# For all images
			for i in range(self.num_images):		
				
				print("#________________________________________________________________#")
				print("Epoch:",e,"Training Image:",i)
				print("#________________________________________________________________#")

				# Intialize the parse tree for this image.=
				self.state = parse_tree_node(label=0,x=0,y=0,w=self.image_size,h=self.image_size)
				self.initialize_tree()
				self.construct_parse_tree(i)	
				self.compute_rewards(i)
				self.propagate_rewards()
				print("Parsing Image:",i)
				print("TOTAL REWARD:",self.parse_tree[0].reward)
				if train:
					self.backprop(i,e)
			if train:
				npy.save("halfparsed_clean3_{0}.npy".format(e),self.predicted_labels)
			else:
				npy.save("validation.npy".format(e),self.predicted_labels)
			self.predicted_labels = npy.zeros((20000,20,20))
			self.save_model(e)

	############################
	# Pixel labels: 
	# 0 for shape
	# 1 for shape with primitive 1
	# 2 for region with no primitive (not to be painted)
	############################

	def map_rules_to_indices(self, rule_index):
		if (rule_index<=3):
			return [0,0]
		if (rule_index==4):
			return 1
		if (rule_index==5):
			return 2

	############################
	# Rule numbers:
	# 0 (Shape) -> (Shape)(Shape) 								(Vertical split)
	# 1 (Shape) -> (Shape)(Shape) 								(Horizontal split)
	# 2 (Shape) -> (Shape)(Shape) 								(Vertical split with opposite order: top-bottom expansion)
	# 3 (Shape) -> (Shape)(Shape) 								(Horizontal split with opposite order: right-left expansion)
	# 4 (Shape) -> (Region with primitive #) 
	# 5 (Shape) -> (Region not to be painted)
	############################

	def preprocess_images_labels(self):
		noise = 0.2*npy.random.rand(self.num_images,self.image_size,self.image_size)
		self.images[npy.where(self.images==2)]=-1
		self.true_labels[npy.where(self.true_labels==2)]=-1
		self.images += noise

def main(args):

	# # Create a TensorFlow session with limits on GPU usage.
	gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list="1,2")
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	hierarchical_model = hierarchical()

	# MUST LOAD IMAGES / LOAD NOISY IMAGES (So that the CNN has some features to latch on to.)	
	hierarchical_model.images = npy.load(str(sys.argv[1]))	
	hierarchical_model.true_labels = npy.load(str(sys.argv[2]))
	
	hierarchical_model.preprocess_images_labels()
	hierarchical_model.plot = 0
	
	load = 0
	if load:
		print("HI!")
		model_file = str(sys.argv[3])
		hierarchical_model.initialize_tensorflow_model(sess,model_file)
	else:
		hierarchical_model.initialize_tensorflow_model(sess)

	# CALL TRAINING
	# hierarchical_model.meta_training(train=False)
	hierarchical_model.meta_training(train=True)

if __name__ == '__main__':
	main(sys.argv)




