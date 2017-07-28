#!/usr/bin/env python
from headers import *
from state_class import *

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
		self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))

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
		if self.image_size == 20:
			self.conv4 = tf.add(tf.nn.conv2d(self.relu_conv3,self.W_conv4,strides=[1,1,1,1],padding='VALID'),self.b_conv4,name='conv4')
		else:
			self.conv4 = tf.add(tf.nn.conv2d(self.relu_conv3,self.W_conv4,strides=[1,2,2,1],padding='VALID'),self.b_conv4,name='conv4')
		self.relu_conv4 = tf.nn.relu(self.conv4)

		# Layer 5
		self.W_conv5 = tf.Variable(tf.truncated_normal([self.conv5_size,self.conv5_size,self.conv4_num_filters,self.conv5_num_filters],stddev=0.1),name='W_conv5')
		self.b_conv5 = tf.Variable(tf.constant(0.1,shape=[self.conv5_num_filters]),name='b_conv5')
		# if self.image_size == 20:
			# self.conv5 = tf.add(tf.nn.conv2d(self.relu_conv4,self.W_conv5,strides=[1,1,1,1],padding='VALID'),self.b_conv5,name='conv5')
			# self.conv5 = tf.add(tf.nn.conv2d(self.relu_conv4,self.W_conv5,strides=[1,2,2,1],padding='VALID'),self.b_conv5,name='conv5')	
		# else:
		self.conv5 = tf.add(tf.nn.conv2d(self.relu_conv4,self.W_conv5,strides=[1,2,2,1],padding='VALID'),self.b_conv5,name='conv5')	
		self.relu_conv5 = tf.nn.relu(self.conv5)

		# Now going to flatten this and move to a fully connected layer.s
		# self.fc_input_shape = 10*10*self.conv5_num_filters
		self.fc_input_shape = 5*5*self.conv5_num_filters
		self.relu_conv5_flat = tf.reshape(self.relu_conv5,[-1,self.fc_input_shape])

		# Going to split into 4 streams: RULE, SPLIT, START and GOAL

		##########################
		# CREATING THE RULE STREAM
		##########################
		self.rulefc_l1_shape = 120
		self.W_rulefc_l1 = tf.Variable(tf.truncated_normal([self.fc_input_shape,self.rulefc_l1_shape],stddev=0.1),name='W_rulefc_l1')
		self.b_rulefc_l1 = tf.Variable(tf.constant(0.1,shape=[self.rulefc_l1_shape]),name='b_rulefc_l1')
		self.rulefc_l1 = tf.nn.relu(tf.add(tf.matmul(self.relu_conv5_flat,self.W_rulefc_l1),self.b_rulefc_l1),name='rulefc_l1')

		# 2nd FC layer: RULE Output:
		self.number_primitives = 1
		self.rulefc_output_shape = 1*self.number_primitives+5
		self.W_rulefc_l2 = tf.Variable(tf.truncated_normal([self.rulefc_l1_shape,self.rulefc_output_shape],stddev=0.1),name='W_rulefc_l2')
		self.b_rulefc_l2 = tf.Variable(tf.constant(0.1,shape=[self.rulefc_output_shape]),name='b_rulefc_l2')
		self.rulefc_presoftmax = tf.add(tf.matmul(self.rulefc_l1,self.W_rulefc_l2),self.b_rulefc_l2,name='rulefc_presoftmax')
		self.rule_probabilities = tf.nn.softmax(self.rulefc_presoftmax,name='softmax')
		
		#########################
		# CREATING A SPLIT STREAM
		#########################
		self.splitfc_l1_shape = 50
		self.W_splitfc_l1 = tf.Variable(tf.truncated_normal([self.fc_input_shape,self.splitfc_l1_shape],stddev=0.1),name='W_splitfc_l1')		
		self.b_splitfc_l1 = tf.Variable(tf.constant(0.1,shape=[self.splitfc_l1_shape]),name='b_splitfc_l1')
		self.splitfc_l1 = tf.nn.relu(tf.add(tf.matmul(self.relu_conv5_flat,self.W_splitfc_l1),self.b_splitfc_l1),name='splitfc_l1')		

		# Split output.
		self.W_split = tf.Variable(tf.truncated_normal([self.splitfc_l1_shape,2],stddev=0.1),name='W_split')
		self.b_split = tf.Variable(tf.constant(0.1,shape=[2]),name='b_split')
		
		self.splitfc_preslice = tf.matmul(self.splitfc_l1,self.W_split)+self.b_split
		self.split_mean = tf.nn.sigmoid(self.splitfc_preslice[0,0])
		self.split_cov = tf.nn.softplus(self.splitfc_preslice[0,1])+0.05
		self.split_dist = tf.contrib.distributions.Normal(loc=self.split_mean,scale=self.split_cov)

		# Sampling a split. Remember, this should still just be defining an operation, not actually sampling.
		self.sample_split = self.split_dist.sample()
		# Also maintaining placeholders for scaling, converting to integer, and back to float.
		self.sampled_split = tf.placeholder(tf.float32,shape=(None),name='sampled_split')

		##########################
		# CREATING THE GOAL STREAM
		##########################
		self.goalfc_l1_shape = 30
		self.W_goalfc_l1 = tf.Variable(tf.truncated_normal([self.fc_input_shape,self.goalfc_l1_shape],stddev=0.1),name='W_goalfc_l1')		
		self.b_goalfc_l1 = tf.Variable(tf.constant(0.1,shape=[self.goalfc_l1_shape]),name='b_goalfc_l1')
		self.goalfc_l1 = tf.nn.relu(tf.add(tf.matmul(self.relu_conv5_flat,self.W_goalfc_l1),self.b_goalfc_l1),name='goalfc_l1')		

		# Goal output.
		self.W_goal = tf.Variable(tf.truncated_normal([self.goalfc_l1_shape,4],stddev=0.1),name='W_goal')
		self.b_goal = tf.Variable(tf.constant(0.1,shape=[4]),name='b_goal')
		self.goalfc_preslice = tf.matmul(self.goalfc_l1,self.W_goal)+self.b_goal
		self.goal_mean = tf.nn.sigmoid(self.goalfc_preslice[0,:2])
		self.goal_cov = tf.nn.softplus(self.goalfc_preslice[0,2:])+0.05

		###########################
		# CREATING THE START STREAM
		###########################
		self.startfc_l1_shape = 30
		self.W_startfc_l1 = tf.Variable(tf.truncated_normal([self.fc_input_shape,self.startfc_l1_shape],stddev=0.1),name='W_startfc_l1')
		self.b_startfc_l1 = tf.Variable(tf.constant(0.1,shape=[self.startfc_l1_shape]),name='b_startfc_l1')
		self.startfc_l1 = tf.nn.relu(tf.add(tf.matmul(self.relu_conv5_flat,self.W_startfc_l1),self.b_startfc_l1),name='startfc_l1')				

		# Start output.
		self.W_start = tf.Variable(tf.truncated_normal([self.startfc_l1_shape,4],stddev=0.1),name='W_start')
		self.b_start = tf.Variable(tf.constant(0.1,shape=[4]),name='b_start')
		self.startfc_preslice = tf.matmul(self.startfc_l1,self.W_start)+self.b_start
		self.start_mean = tf.nn.sigmoid(self.startfc_preslice[0,:2])
		self.start_cov = tf.nn.softplus(self.startfc_preslice[0,2:])+0.05

		# Creating start and goal distributions.
		self.goal_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.goal_mean,scale_diag=self.goal_cov)
		self.start_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.start_mean,scale_diag=self.start_cov)

		self.sample_goal = self.goal_dist.sample()
		self.sample_start = self.start_dist.sample()

		self.sampled_goal = tf.placeholder(tf.float32,shape=(2),name='sampled_goal')
		self.sampled_start = tf.placeholder(tf.float32,shape=(2),name='sampled_start')
		self.previous_goal = tf.placeholder(tf.float32,shape=(2),name='previous_goal')

		# Defining training ops. 
		self.rule_return_weight = tf.placeholder(tf.float32,shape=(None),name='rule_return_weight')
		self.split_return_weight = tf.placeholder(tf.float32,shape=(None),name='split_return_weight')
		self.startgoal_return_weight = tf.placeholder(tf.float32,shape=(None),name='startgoal_return_weight')
		self.target_rule = tf.placeholder(tf.float32,shape=( self.rulefc_output_shape),name='target_rule')

		# Defining the loss for each of the 3 streams, rule, split and goal.
		# Rule loss is the negative cross entropy between the rule probabilities and the chosen rule as a one-hot encoded vector. 
		# Weighted by the return obtained. This is just the negative log probability of the selected action.

		# NO NEGATIVE SIGN HERE
		self.rule_loss = tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=self.target_rule,logits=self.rulefc_presoftmax),self.rule_return_weight) 
		# The split loss is the negative log probability of the chosen split, weighted by the return obtained.
		self.split_loss = -tf.multiply(self.split_dist.log_prob(self.sampled_split),self.split_return_weight)
		# The goal loss is the negative log probability of the chosen goal, weighted by the return obtained.
		self.goal_loss = -tf.multiply(self.goal_dist.log_prob(self.sampled_goal),self.startgoal_return_weight)
		# The start loss is the negative log probability of the chosen start, weighted byt he return obtained.
		self.start_loss = -tf.multiply(self.start_dist.log_prob(self.sampled_start),self.startgoal_return_weight)
		# The total loss is the sum of individual losses.
		self.total_loss = self.rule_loss + self.split_loss + self.start_loss + self.goal_loss

		# Creating a training operation to minimize the total loss.
		self.train = tf.train.AdamOptimizer(1e-4).minimize(self.total_loss,name='Adam_Optimizer')

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
		if not(os.path.isdir("saved_models")):
			os.mkdir("saved_models")
		save_path = self.saver.save(self.sess,'saved_models/net6_conv5_model_{0}.ckpt'.format(model_index))

	def initialize_tree(self):
		# Intialize the parse tree for this image.=
		self.state = parse_tree_node(label=0,x=0,y=0,w=self.image_size,h=self.image_size)
		self.current_parsing_index = 0
		self.parse_tree = [parse_tree_node()]
		self.parse_tree[self.current_parsing_index]=self.state

	def insert_node(self, state, index):
		# Insert node into the parse tree.
		self.parse_tree.insert(index,state)

	def parse_nonterminal(self, image_index):
		rule_probabilities = self.sess.run(self.rule_probabilities,feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
	
		# THIS IS THE RULE POLICY: This is a probabilistic selection of the rule., completely random.
		# Should it be an epsilon-greedy policy? 

		# SAMPLING A SPLIT LOCATION
		split_location = -1

		# Hard coding ban of vertical splits when h==1, and of horizontal splits when w==1.
		# CHANGING THIS NOW TO BAN SPLITS FOR REGIONS SMALLER THAN: MINIMUM_WIDTH; and not just if ==1.
		self.minimum_width = 3
		# print(rule_probabilities[0])
		
		epislon = 1e-5
		rule_probabilities += epislon

		if (self.state.h<=self.minimum_width):
			rule_probabilities[0][[0,2]]=0.

		if (self.state.w<=self.minimum_width):
			rule_probabilities[0][[1,3]]=0.

		# print(rule_probabilities[0])

		rule_probabilities/=rule_probabilities.sum()
		selected_rule = npy.random.choice(range(self.rulefc_output_shape),p=rule_probabilities[0])
		indices = self.map_rules_to_indices(selected_rule)

		# print("Selected Rule:",selected_rule)

		# If it is a split rule:
		if selected_rule<=3:

			# Resampling until it gets a split INSIDE the segment.
			# This just ensures the split lies within 0 and 1.
			# while (split_location<=0)or(split_location>=1):

			# Apply the rule: if the rule number is even, it is a vertical split and if the current non-terminal to be parsed is taller than 1 unit:
			# if (selected_rule%2==0) and (self.state.h>1):
			# if (selected_rule==0):		
			if ((selected_rule==0) or (selected_rule==2)):
				counter = 0				

				while (split_location<=0)or(split_location>=self.state.h):
					split_location = self.sess.run(self.sample_split, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
					counter += 1

					split_copy = split_location
					inter_split = split_location*self.state.h

					if inter_split>(self.image_size/2):
						split_location = int(npy.floor(inter_split))
					else:
						split_location = int(npy.ceil(inter_split))

				# # SAMPLING SPLIT LOCATION INSIDE THIS CONDITION:
				# while (int(self.state.h*split_location)<=0)or(int(self.state.h*split_location)>=self.state.h):
				# 	split_location = self.sess.run(self.sample_split, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
				# 	counter+=1

				# 	if counter>25:
				# 		print("State: H",self.state.h)
				# 		print("Split fraction:",split_location)
				# 		print("Split location:",int(split_location*self.state.h))

				# split_copy = copy.deepcopy(split_location)
				# split_location = int(self.state.h*split_location)
			
				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=self.state.w,h=split_location,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x,y=self.state.y+split_location,w=self.state.w,h=self.state.h-split_location,backward_index=self.current_parsing_index)

			# if (selected_rule==1):			
			if ((selected_rule==1) or (selected_rule==3)):
				counter = 0

				while (split_location<=0)or(split_location>=self.state.w):
					split_location = self.sess.run(self.sample_split, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
					counter += 1

					split_copy = split_location
					inter_split = split_location*self.state.w

					if inter_split>(self.image_size/2):
						split_location = int(npy.floor(inter_split))
					else:
						split_location = int(npy.ceil(inter_split))

				# # SAMPLING SPLIT LOCATION INSIDE THIS CONDITION:
				# while (int(self.state.w*split_location)<=0)or(int(self.state.w*split_location)>=self.state.w):
				# 	split_location = self.sess.run(self.sample_split, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
				# 	counter+=1
				# 	if counter>25:
				# 		print("State: W",self.state.w)
				# 		print("Split fraction:",split_location)
				# 		print("Split location:",int(split_location*self.state.w))

				# # Scale split location.
				# split_copy = copy.deepcopy(split_location)
				# split_location = int(self.state.w*split_location)

				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=split_location,h=self.state.h,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x+split_location,y=self.state.y,w=self.state.w-split_location,h=self.state.h,backward_index=self.current_parsing_index)
				
			# Update current parse tree with split location and rule applied.
			self.parse_tree[self.current_parsing_index].split=split_copy
			self.parse_tree[self.current_parsing_index].rule_applied=selected_rule

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

	def parse_terminal(self, image_index):
		
		# If terminal with primitive assigned:
		if (self.state.label==1):
			
			# Sample a goal location and start location.
			start_location, goal_location, start_mean, goal_mean = self.sess.run([self.sample_start,self.sample_goal, \
				self.start_mean, self.goal_mean],feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
			
			start_copy = copy.deepcopy(start_location)
			goal_copy = copy.deepcopy(goal_location)

 			start_location *= npy.array([self.state.w,self.state.h])
			goal_location *= npy.array([self.state.w,self.state.h])

			# print("START / GOAL: ")
			# print(start_copy, start_location, goal_copy, goal_location)

			self.parse_tree[self.current_parsing_index].start = npy.array(start_location)
			self.parse_tree[self.current_parsing_index].goal = npy.array(goal_location)
			
			self.state = self.parse_tree[self.current_parsing_index]

			# Compute the angle and the length of the start-goal line. 
			angle = npy.arctan2((self.state.goal[1]-self.state.start[1]),(self.state.goal[0]-self.state.start[0]))
			length = npy.linalg.norm(self.state.goal-self.state.start)

			# Create a rectangle for the paintbrush.
			rect = box(self.state.start[0]+self.state.x,self.state.y+self.state.start[1]-self.paintwidth,self.state.x+self.state.start[0]+length,self.state.y+self.state.start[1]+self.paintwidth)		

			# Rotate it.	
			rotated_rect = rotate(rect,angle,origin=[self.state.x+self.state.start[0],self.state.y+self.state.start[1]],use_radians=True)

			coords = npy.array(list(rotated_rect.exterior.coords)[0:4])
			bounding_height = int(npy.ceil(max(self.state.y+self.state.h,npy.max(coords[:,1]))))
			bounding_width = int(npy.ceil(max(self.state.x+self.state.w,npy.max(coords[:,0]))))

			for x in range(int(self.state.x),bounding_width):
				for y in range(int(self.state.y), bounding_height):
					if rotated_rect.contains(point.Point(x,y)) and (x<self.image_size) and (y<self.image_size):
						self.painted_image[x,y] = 1
						self.painted_images[image_index,x,y] = 1

			self.parse_tree[self.current_parsing_index].start = start_copy
			self.parse_tree[self.current_parsing_index].goal = goal_copy

		self.state.reward = (self.true_labels[image_index, self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]*self.painted_image[self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]).sum()
		# To make sure parse tree is being assigned the reward value in case 
		# state is creating a copy of the parse_tree_node object rather than referencing.
		self.parse_tree[self.current_parsing_index].reward = self.state.reward
		self.current_parsing_index+=1

	def backprop(self, image_index):
		# Must decide whether to do this stochastically or in batches. # For now, do it stochastically, moving forwards through the tree.

		# NOW CHANGING TO 4 RULE SYSTEM.
		target_rule = npy.zeros(self.rulefc_output_shape)

		for j in range(len(self.parse_tree)):
			self.state = self.parse_tree[j]
			
			boundary_width = 2
			lowerx = max(0,self.state.x-boundary_width)
			upperx = min(self.image_size,self.state.x+self.state.w+boundary_width)
			lowery = max(0,self.state.y-boundary_width)
			uppery = min(self.image_size,self.state.y+self.state.h+boundary_width)

			self.image_input = self.images[image_index, lowerx:upperx, lowery:uppery]
			self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))

			rule_weight = 0.
			split_weight = 0.
			startgoal_weight = 0.
			target_rule = npy.zeros(self.rulefc_output_shape)

			if self.parse_tree[j].label==0:
				
				rule_weight = self.parse_tree[j].reward
				target_rule[self.parse_tree[j].rule_applied] = 1.

				# Nested, since a split rule may only be applied on a non-terminal.	
				if self.parse_tree[j].rule_applied<=3:
					split_weight = self.parse_tree[j].reward				

			if self.parse_tree[j].label==1:
				startgoal_weight = self.parse_tree[j].reward

			# Here ,we only backprop for shapes, since we only choose actions for shapese.
			rule_loss, split_loss, _ = self.sess.run([self.rule_loss, self.split_loss, self.train], \
				feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1), \
					 self.rule_return_weight: rule_weight, self.split_return_weight: split_weight, self.target_rule: target_rule, \
						 self.sampled_goal: self.parse_tree[j].goal, self.sampled_start: self.parse_tree[j].start, \
						 	self.startgoal_return_weight: startgoal_weight, self.sampled_split: self.parse_tree[j].split})

			# print("LOSS VALUES:",rule_loss, split_loss)

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

			# If the current non-terminal is a shape.
			if (self.state.label==0):
				# print("PARSING NON TERMINAL")
				self.parse_nonterminal(image_index)

			# If the current non-terminal is a region assigned a particular primitive.
			if (self.state.label==1) or (self.state.label==2):	
				self.parse_terminal(image_index)
				
			# Update data for plots
			# self.alternate_painted_image[npy.where(self.predicted_labels[image_index]==1)]=1.			
			self.alternate_painted_image[npy.where(self.painted_image==1)]=1.
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
	
	def meta_training(self,train=True):

		image_index = 0
		self.painted_image = -npy.ones((self.image_size,self.image_size))

		self.define_plots()

		# For all epochs
		if not(train):
			self.num_epochs=1

		# For all epochs
		for e in range(self.num_epochs):
			for i in range(self.num_images):

				self.initialize_tree()
				self.construct_parse_tree(i)				
				# Rewards are assigned while parsing terminal symbols in this case;
				# No need to separately compute the rewards; only propagate them. 
				# self.compute_rewards(i)
				self.propagate_rewards()
				print("#___________________________________________________________________________")
				print("Epoch:",e,"Training Image:",i,"TOTAL REWARD:",self.parse_tree[0].reward)

				if train:
					self.backprop(i)

			if train:
				npy.save("parsed_{0}.npy".format(e),self.predicted_labels)
				self.save_model(e)
			else: 
				npy.save("validation.npy".format(e),self.predicted_labels)

			self.predicted_labels = npy.zeros((self.num_images,self.image_size,self.image_size))
			self.painted_images = npy.zeros((self.num_images,self.image_size,self.image_size))

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
	hierarchical_model.initialize_tensorflow_model(sess)

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
