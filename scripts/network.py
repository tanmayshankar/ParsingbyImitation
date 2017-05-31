#!/usr/bin/env python

from headers import *
# Define a class for the parse tree / rule / etc? 
class parse_tree_node():
	def __init__(self, label=-1, x=-1, y=-1,w=-1,h=-1,backward_index=-1,rule_applied=-1, split=-1, start=npy.array([-1,-1]), goal=npy.array([-1,-1])):
		self.label = label
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.backward_index = backward_index
		self.rule_applied = rule_applied
		self.split = split
		self.start = start
		self.goal = goal
		self.reward = 0.

	def disp(self):
		print("Label:", self.label)
		print("X:",self.x,"Y:",self.y,"W:",self.w,"H:",self.h)

class hierarchical():

	def __init__(self):

		self.num_epochs = 2
		self.num_images = 1000

		self.current_parsing_index = 0
		self.parse_tree = [parse_tree_node()]
		self.paintwidth=2
		self.images = []
		self.true_labels = []
		self.image_size = 20
		self.predicted_labels = npy.zeros((self.num_images,self.image_size, self.image_size))

	def initialize_tensorflow_model(self, sess):

		# Initializing the session.
		self.sess = sess

		# Image size and other architectural parameters. 
		
		self.conv1_size = 3	
		self.conv1_num_filters = 20

		self.conv2_size = 3	
		self.conv2_num_filters = 20

		self.conv3_size = 3	
		self.conv3_num_filters = 20

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

		# Now going to flatten this and move to a fully connected layer.s
		self.fc_input_shape = 14*14*self.conv3_num_filters
		self.relu_conv3_flat = tf.reshape(self.relu_conv3,[-1,self.fc_input_shape])

		# Going to split into 3 streams: RULE, SPLIT, and GOAL
		self.fcs1_l1_shape = 120
		self.W_fcs1_l1 = tf.Variable(tf.truncated_normal([self.fc_input_shape,self.fcs1_l1_shape],stddev=0.1),name='W_fcs1_l1')
		self.b_fcs1_l1 = tf.Variable(tf.constant(0.1,shape=[self.fcs1_l1_shape]),name='b_fcs1_l1')
		self.fcs1_l1 = tf.nn.relu(tf.add(tf.matmul(self.relu_conv3_flat,self.W_fcs1_l1),self.b_fcs1_l1),name='fcs1_l1')

		self.fcs2_l1_shape = 30
		self.W_fcs2_l1 = tf.Variable(tf.truncated_normal([self.fc_input_shape,self.fcs2_l1_shape],stddev=0.1),name='W_fcs2_l1')		
		self.b_fcs2_l1 = tf.Variable(tf.constant(0.1,shape=[self.fcs2_l1_shape]),name='b_fcs2_l1')
		self.fcs2_l1 = tf.nn.relu(tf.add(tf.matmul(self.relu_conv3_flat,self.W_fcs2_l1),self.b_fcs2_l1),name='fcs2_l1')		

		self.fcs3_l1_shape = 30
		self.W_fcs3_l1 = tf.Variable(tf.truncated_normal([self.fc_input_shape,self.fcs3_l1_shape],stddev=0.1),name='W_fcs3_l1')		
		self.b_fcs3_l1 = tf.Variable(tf.constant(0.1,shape=[self.fcs3_l1_shape]),name='b_fcs3_l1')
		self.fcs3_l1 = tf.nn.relu(tf.add(tf.matmul(self.relu_conv3_flat,self.W_fcs3_l1),self.b_fcs3_l1),name='fcs3_l1')		

		self.fcs4_l1_shape = 30
		self.W_fcs4_l1 = tf.Variable(tf.truncated_normal([self.fc_input_shape,self.fcs4_l1_shape],stddev=0.1),name='W_fcs4_l1')
		self.b_fcs4_l1 = tf.Variable(tf.constant(0.1,shape=[self.fcs4_l1_shape]),name='b_fcs4_l1')
		self.fcs4_l1 = tf.nn.relu(tf.add(tf.matmul(self.relu_conv3_flat,self.W_fcs4_l1),self.b_fcs4_l1),name='fcs4_l1')				

		# 2nd FC layer: Output:
		self.number_primitives = 1
		self.fcs1_output_shape = 5*self.number_primitives+3
		self.W_fcs1_l2 = tf.Variable(tf.truncated_normal([self.fcs1_l1_shape,self.fcs1_output_shape],stddev=0.1),name='W_fcs1_l2')
		self.b_fcs1_l2 = tf.Variable(tf.constant(0.1,shape=[self.fcs1_output_shape]),name='b_fcs1_l2')
		self.fcs1_presoftmax = tf.add(tf.matmul(self.fcs1_l1,self.W_fcs1_l2),self.b_fcs1_l2,name='fcs1_presoftmax')
		self.rule_probabilities = tf.nn.softmax(self.fcs1_presoftmax,name='softmax')
		print(self.rule_probabilities)
		# Split output.
		self.W_split = tf.Variable(tf.truncated_normal([self.fcs2_l1_shape,2],stddev=0.1),name='W_split')
		self.b_split = tf.Variable(tf.constant(0.1,shape=[2]),name='b_split')
		
		self.fcs2_preslice = tf.matmul(self.fcs2_l1,self.W_split)+self.b_split
		self.split_mean = tf.nn.sigmoid(self.fcs2_preslice[0,0])
		self.split_cov = tf.nn.relu(self.fcs2_preslice[0,1])

		# Goal output.
		self.W_goal = tf.Variable(tf.truncated_normal([self.fcs3_l1_shape,4],stddev=0.1),name='W_goal')
		self.b_goal = tf.Variable(tf.constant(0.1,shape=[4]),name='b_goal')
		
		self.fcs3_preslice = tf.matmul(self.fcs3_l1,self.W_goal)+self.b_goal
		# self.goal_mean = tf.nn.sigmoid(self.fcs3_preslice[:2])
		# self.goal_cov = tf.nn.relu(self.fcs3_preslice[2:])		

		self.goal_mean = tf.nn.sigmoid(self.fcs3_preslice[0,:2])
		self.goal_cov = tf.nn.relu(self.fcs3_preslice[0,2:])		

		# Start output.
		self.W_start = tf.Variable(tf.truncated_normal([self.fcs3_l1_shape,4],stddev=0.1),name='W_start')
		self.b_start = tf.Variable(tf.constant(0.1,shape=[4]),name='b_start')
		
		self.fcs3_preslice = tf.matmul(self.fcs3_l1,self.W_start)+self.b_start
		self.start_mean = tf.nn.sigmoid(self.fcs3_preslice[0,:2])
		self.start_cov = tf.nn.relu(self.fcs3_preslice[0,2:])		

		# CONSTRUCTING THE DISTRIBUTIONS FOR GOALS AND SPLITS
		self.goal_dist = tf.contrib.distributions.MultivariateNormalDiag(self.goal_mean,self.goal_cov)
		self.start_dist = tf.contrib.distributions.MultivariateNormalDiag(self.start_mean,self.start_cov)
		self.split_dist = tf.contrib.distributions.Normal(mu=self.split_mean,sigma=self.split_cov)

		# self.sample_goal = tf.placeholder(tf.float32,shape=(None,2),name='sample_goal')
		# self.sample_split = tf.placeholder(tf.float32,shape=(None,1),name='sample_split')

		# Sampling a goal and a split. Remember, this should still just be defining an operation, not actually sampling.
		# We evaluate this to retrieve a sample goal / split location. 
		self.sample_split = self.split_dist.sample()
		self.sample_goal = self.goal_dist.sample()
		self.sample_start = self.start_dist.sample()

		# Also maintaining placeholders for scaling, converting to integer, and back to float.
		# self.sampled_split = tf.placeholder(tf.float32,shape=(None,1),name='sampled_split')
		# self.sampled_goal = tf.placeholder(tf.float32,shape=(None,2),name='sampled_goal')
		# self.sampled_start = tf.placeholder(tf.float32,shape=(None,2),name='sampled_start')
		# self.previous_goal = tf.placeholder(tf.float32,shape=(None,2),name='previous_goal')

		self.sampled_split = tf.placeholder(tf.float32,shape=(None),name='sampled_split')
		self.sampled_goal = tf.placeholder(tf.float32,shape=(2),name='sampled_goal')
		self.sampled_start = tf.placeholder(tf.float32,shape=(2),name='sampled_start')
		self.previous_goal = tf.placeholder(tf.float32,shape=(2),name='previous_goal')

		# # # # # Defining training ops. 
		self.rule_return_weight = tf.placeholder(tf.float32,shape=(None),name='rule_return_weight')
		self.split_return_weight = tf.placeholder(tf.float32,shape=(None),name='split_return_weight')
		# self.goal_return_weight = tf.placeholder(t.ffloat32,shape=( 1),name='goal_return_weight')
		# self.start_return_weight = tf.placeholder(tf.float32,shape=( 1),name='start_return_weight')
		self.startgoal_return_weight = tf.placeholder(tf.float32,shape=(None),name='startgoal_return_weight')

		self.target_rule = tf.placeholder(tf.float32,shape=( self.fcs1_output_shape),name='target_rule')

		# Defining the loss for each of the 3 streams, rule, split and goal.
		# Rule loss is the negative cross entropy between the rule probabilities and the chosen rule as a one-hot encoded vector. 
		# Weighted by the return obtained. This is just the negative log probability of the selected action.
		self.rule_loss = -tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=self.target_rule,logits=self.fcs1_presoftmax),self.rule_return_weight)

		# The split loss is the negative log probability of the chosen split, weighted by the return obtained.
		self.split_loss = -tf.multiply(self.split_dist.log_prob(self.sampled_split),self.split_return_weight)

		# The goal loss is the negative log probability of the chosen goal, weighted by the return obtained.
		self.goal_loss = -tf.multiply(self.goal_dist.log_prob(self.sampled_goal),self.startgoal_return_weight)

		# The start loss is the negative log probability of the chosen start, weighted byt he return obtained.
		self.start_loss = -tf.multiply(self.start_dist.log_prob(self.sampled_start),self.startgoal_return_weight)

		# Start goal loss - difference between current starting point and previous goal location - this must be in GLOBAL COORDINATES
		self.startgoal_loss = tf.multiply(tf.square(tf.norm(tf.subtract(self.previous_goal,self.sampled_start))),self.startgoal_return_weight)

		# The total loss is the sum of individual losses.
		self.total_loss = self.rule_loss + self.split_loss + self.goal_loss + self.startgoal_loss + self.start_loss

		# Creating a training operation to minimize the total loss.
		self.train = tf.train.AdamOptimizer(1e-4).minimize(self.total_loss,name='Adam_Optimizer')

		init = tf.global_variables_initializer()
		self.sess.run(init)

	def initialize_tree(self):
		self.current_parsing_index = 0
		self.parse_tree[self.current_parsing_index]=self.state

	def insert_node(self, state, index):
		self.parse_tree.insert(index,state)

	def parse_nonterminal(self, image_index):
		rule_probabilities = self.sess.run(self.rule_probabilities,feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
	
		# THIS IS THE RULE POLICY: This is a probabilistic selection of the rule., completely random.
		# Should it be an epsilon-greedy policy? 

		selected_rule = npy.random.choice(range(self.fcs1_output_shape),p=rule_probabilities[0])
		indices = self.map_rules_to_indices(selected_rule)
		
		# If rules 0-5, need a split location.
		# If 6, need a goal location.

		# SAMPLING A SPLIT LOCATION
	
		split_location = -1

		# In case the selected rule is a vertical split when height is 1, or horizontal split when width is 1.
		# if ((selected_rule%2==0) and (self.state.h==1)) or ((selected_rule%2!=0)and(self.state.w==1)):
	
		# Hard coding ban of vertical splits when h==1, and of horizontal splits when w==1.
		if (self.state.h==1):
			rule_probabilities[0][[0,2,4]]=0.
		if (self.state.w==1):
			rule_probabilities[0][[1,3,5]]=0.
		rule_probabilities/=rule_probabilities.sum()
		selected_rule = npy.random.choice(range(self.fcs1_output_shape),p=rule_probabilities[0])
		indices = self.map_rules_to_indices(selected_rule)
		print("Selected Rule:",selected_rule)

		if selected_rule<=5:
			
			# Resampling until it gets a split INSIDE the segment.
			# This just ensures the split lies within 0 and 1.
			# while (split_location<=0)or(split_location>=1):

			# # while (int(split_location*self.state.h)<=0)or(int(split_location*self.state.w)>=1):
			# while ((int(self.state.h*split_location)<=0)or(int(self.state.h*split_location)>=self.state.h))or((int(self.state.w*split_location)<=0)or(int(self.state.w*split_location)>=self.state.w)):
			# 	split_location = self.sess.run(self.sample_split, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
			
			# Apply the rule: if the rule number is even, it is a vertical split and if the current non-terminal to be parsed is taller than 1 unit:
			if (selected_rule%2==0) and (self.state.h>1):
				
				# SAMPLING SPLIT LOCATION INSIDE THIS CONDITION:
				while (int(self.state.h*split_location)<=0)or(int(self.state.h*split_location>=self.state.h)):
					split_location = self.sess.run(self.sample_split, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
				print("SPLIT LOCATION:",split_location)
				# Scale split location.		
				split_location = int(self.state.h*split_location)
				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=self.state.w,h=split_location,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x,y=self.state.y+split_location,w=self.state.w,h=self.state.h-split_location,backward_index=self.current_parsing_index)
				
				# Update current parse tree with split location and rule applied.
				self.parse_tree[self.current_parsing_index].split=split_location
				self.parse_tree[self.current_parsing_index].rule_applied = selected_rule

				# Insert splits into parse tree.
				self.insert_node(s1,self.current_parsing_index+1)
				self.insert_node(s2,self.current_parsing_index+2)
				self.current_parsing_index+=1

				# Modify the images.
				# self.images[image_index,s1.x:s1.x+s1.w,s1.y:s1.y+s1.h] = s1.label
				# self.images[image_index,s2.x:s2.x+s2.w,s2.y:s2.y+s2.h] = s2.label

				self.predicted_labels[image_index,s1.x:s1.x+s1.w,s1.y:s1.y+s1.h] = s1.label
				self.predicted_labels[image_index,s2.x:s2.x+s2.w,s2.y:s2.y+s2.h] = s2.label

			if (selected_rule%2!=0) and (self.state.w>1):

				# SAMPLING SPLIT LOCATION INSIDE THIS CONDITION:
				while (int(self.state.w*split_location)<=0)or(int(self.state.w*split_location>=self.state.w)):
					split_location = self.sess.run(self.sample_split, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
				print("SPLIT LOCATION:",split_location)
				# Scale split location.
				split_location = int(self.state.w*split_location)
				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=split_location,h=self.state.h,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x+split_location,y=self.state.y,w=self.state.w-split_location,h=self.state.h,backward_index=self.current_parsing_index)

				# Update current parse tree with split location and rule applied.
				self.parse_tree[self.current_parsing_index].split=split_location
				self.parse_tree[self.current_parsing_index].rule_applied = selected_rule

				# Insert splits into parse tree.
				self.insert_node(s1,self.current_parsing_index+1)
				self.insert_node(s2,self.current_parsing_index+2)
				self.current_parsing_index+=1

				# # Modify the images.
				# self.images[image_index,s1.x:s1.x+s1.w,s1.y:s1.y+s1.h] = s1.label
				# self.images[image_index,s2.x:s2.x+s2.w,s2.y:s2.y+s2.h] = s2.label		

				self.predicted_labels[image_index,s1.x:s1.x+s1.w,s1.y:s1.y+s1.h] = s1.label
				self.predicted_labels[image_index,s2.x:s2.x+s2.w,s2.y:s2.y+s2.h] = s2.label

		elif selected_rule>=6:

			# Create a parse tree node object.
			s1 = copy.deepcopy(self.parse_tree[self.current_parsing_index])
			# Change label.
			s1.label=selected_rule-5
			# Change the backward index.
			s1.backward_index = self.current_parsing_index

			# Update current parse tree with rule applied.
			self.parse_tree[self.current_parsing_index].rule_applied = selected_rule

			# Insert node into parse tree.
			self.insert_node(s1,self.current_parsing_index+1)
			self.current_parsing_index+=1						
			# self.images[image_index,s1.x:s1.x+s1.w,s1.y:s1.y+s1.h] = s1.label
			self.predicted_labels[image_index,s1.x:s1.x+s1.w,s1.y:s1.y+s1.h] = s1.label

	def parse_primitive_terminal(self):
		# Sample a goal location.
		start_location, goal_location = self.sess.run([self.sample_start,self.sample_goal],feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1)})
		start_location *= npy.array([self.state.w,self.state.h])
		goal_location *= npy.array([self.state.w,self.state.h])
		self.parse_tree[self.current_parsing_index].goal = npy.array(goal_location)
		self.parse_tree[self.current_parsing_index].start = npy.array(start_location)

		self.current_parsing_index+=1
	def propagate_rewards(self):

		# Traverse the tree in reverse order, accumulate rewards into parent nodes recursively as sum of rewards of children.
		# This is actually the return accumulated by any particular decision.
		for j in reversed(range(len(self.parse_tree))):	
			self.parse_tree[self.parse_tree[j].backward_index].reward += self.parse_tree[j].reward

	# def paint_image(self):
	# 	# DEFINES JUST THE COVERAGE OBJECTIVE:

	# 	# Compute the angle and the length of start-goal line.
	# 	angle = npy.arctan2((self.state.goal[1]-self.state.start[1]),(self.state.goal[0]-self.state.start[0]))
	# 	length = npy.linalg.norm(self.state.goal-self.state.start)

	# 	# # Create a rectangle from the paint brush. - This was in the local frame.
	# 	# rect = box(self.state.start[0],self.state.start[1]-self.paintwidth,self.state.start[0]+length,self.state.start[1]+self.paintwidth)

	# 	# Creating the rectangle in the global frame.
	# 	rect = box(self.state.start[0]+self.state.x,self.state.y+self.state.start[1]-self.paintwidth,self.state.x+self.state.start[0]+length,self.state.y+self.state.start[1]+self.paintwidth)		
	# 	# Rotate it.
	# 	rotated_rect = rotate(rect,angle,origin=[self.state.x+self.state.start[0],self.state.y+self.state.start[1]])

	# 	for x in range(self.image_size):
	# 		for y in range(self.image_size):
	# 			if rotated_rect.contains(point.Point(x,y)):
	# 				self.painted_image[x,y] = 1

	# def terminal_reward(self):
	# 	# Compute the angle and the length of the start-goal line. 
	# 	angle = npy.artctan2((self.state.goal[1]-self.state.start[1]),(self.state.goal[0]-self.state.start[0]))
	# 	length = npy.linalg.norm(self.state.goal-self.state.start)

	# 	# Create a rectangle for the paintbrush.
	# 	rect = box(self.state.start[0]+self.state.x,self.state.y+self.state.start[1]-self.paintwidth,self.state.x+self.state.start[0]+length,self.state.y+self.state.start[1]+self.paintwidth)		
	# 	# Rotate it.
	# 	rotated_rect = rotate(rect,angle,origin=[self.state.x+self.state.start[0],self.state.y+self.state.start[1]])

	# 	for x in range(self.state.x,self.state.x+self.state.w):
	# 		for y in range(self.state.y,self.state.y+self.state.h):
	# 			if rotated_rect.contains(point.Point(x,y)):
	# 				self.painted_image[x,y] = 1

	def terminal_reward(self, image_index):
		# Compute the angle and the length of the start-goal line. 
		angle = npy.arctan2((self.state.goal[1]-self.state.start[1]),(self.state.goal[0]-self.state.start[0]))
		print(self.state.goal,self.state.start)		
		length = npy.linalg.norm(self.state.goal-self.state.start)

		# Create a rectangle for the paintbrush.
		rect = box(self.state.start[0]+self.state.x,self.state.y+self.state.start[1]-self.paintwidth,self.state.x+self.state.start[0]+length,self.state.y+self.state.start[1]+self.paintwidth)		
		# Rotate it.
		rotated_rect = rotate(rect,angle,origin=[self.state.x+self.state.start[0],self.state.y+self.state.start[1]])

		coords = npy.array(list(rotated_rect.exterior.coords))
		bounding_height = int(npy.ceil(max(self.state.y+self.state.h,npy.max(coords[:,1]))))
		bounding_width = int(npy.ceil(max(self.state.x+self.state.w,npy.max(coords[:,0]))))

		bounding_rect = box(self.state.x,self.state.y,bounding_width,bounding_height)		

		for x in range(int(self.state.x),bounding_width):
			for y in range(int(self.state.y), bounding_height):
		# for x in range(self.state.x,self.state.x+self.state.w):
		# 	for y in range(self.state.y,self.state.y+self.state.h):

				if rotated_rect.contains(point.Point(x,y)) and (x<self.image_size) and (y<self.image_size):
					self.painted_image[x,y] = 1

		# self.state.reward = (self.images[image_index, self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]*self.painted_image[self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]).sum()
		self.state.reward = (self.true_labels[image_index, self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]*self.painted_image[self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]).sum()

	def compute_rewards(self, image_index):
		# For all terminal symbols only.

		# Rectange intersection
		# self.painted_image = npy.zeros((self.image_size,self.image_size))
		self.painted_image = -npy.ones((self.image_size,self.image_size))
	
		for j in range(len(self.parse_tree)):

			# Assign state.
			self.state = self.parse_tree[j]

			# For every node in the tree, we know the ground truth image labels.
			# We will compute the reward as:
			# To be painted (-1 for no, 1 for yes)
			# Whether it was painted (-1 for no or 1 for yes)

			# If it is a region with a primitive.
			if self.parse_tree[j].label==1:
				# self.paint_image()
				self.terminal_reward(image_index)

	def backprop(self, image_index):
		# Must decide whether to do this stochastically or in batches.

		# For now, do it stochastically, moving forwards through the tree.
		start = npy.zeros(2)
		previous_goal = npy.zeros(2)
		goal = npy.zeros(2)
		target_rule = npy.zeros(self.fcs1_output_shape)

		for j in range(len(self.parse_tree)):

			self.state = self.parse_tree[self.current_parsing_index]
			# Pick up correct portion of image.
			self.image_input = self.images[image_index, self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]
			self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))

			rule_weight = 0
			startgoal_weight = 0
			split_weight = 0
			target_rule = npy.zeros(self.fcs1_output_shape)
			# MUST PARSE EVERY NODE

			# If shape:
			if self.parse_tree[j].label==0:
				# If split rule.
				if self.parse_tree[j].rule_applied<=5:
					split_weight = self.parse_tree[j].reward
					rule_weight = self.parse_tree[j].reward
					target_rule[self.parse_tree[j].rule_applied] = 1.
				# If rule 6.
				if self.parse_tree[j].rule_applied==6:
					rule_weight = self.parse_tree[j].reward

			# If region with primitive:
			if self.parse_tree[j].label==1:
				startgoal_weight = self.parse_tree[j].reward

			# RUN TRAIN
			print("PRINT split:",self.parse_tree[j].split)
			rule_loss, split_loss, start_loss, goal_loss, startgoal_loss, _ = self.sess.run([self.rule_loss, self.split_loss, self.start_loss,self.goal_loss,self.startgoal_loss, self.train], \
				feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1), self.sampled_split: self.parse_tree[j].split, self.sampled_goal: self.parse_tree[j].goal, self.sampled_start: self.parse_tree[j].start, \
							self.previous_goal: previous_goal, self.rule_return_weight: rule_weight, self.split_return_weight: split_weight, self.startgoal_return_weight: startgoal_weight, self.target_rule: target_rule})

			previous_goal = goal.copy()

	def construct_parse_tree(self,image_index):
		# WHILE WE TERMINATE THAT PARSE:
		# while ((self.predicted_labels[image_index]==0).any()):
		while ((self.predicted_labels[image_index]==0).any() or (self.current_parsing_index<=len(self.parse_tree)-1)):
			# Forward pass of the rule policy- basically picking which rule.
			self.state = self.parse_tree[self.current_parsing_index]
			# Pick up correct portion of image.
			self.image_input = self.images[image_index, self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]
			print("Parsing the following state:")
			self.state.disp()
			print("IMAGE INPUT SHAPE:",self.image_input.shape)
			self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))

			# If the current non-terminal is a shape.
			if (self.state.label==0):
				print("PARSING NON TERMINAL")
				self.parse_nonterminal(image_index)

			# If the current non-terminal is a region assigned a particular primitive.
			# if (state==1)or(state==2)or(state==3)or(state==4):
			if (self.state.label==1):
				# print("________  PARSING TERMINAL")
				self.parse_primitive_terminal()
			print("_______________________")
			print("PRINTING THE PARSE TREE:")
			print("Length of tree:",len(self.parse_tree))
			print("Current:",self.current_parsing_index)
			if (self.state.label==2):
				self.current_parsing_index+=1

			for j in range(len(self.parse_tree)):
				print("Printing Node",j)
				self.parse_tree[j].disp()
				# print(self.parse_tree[j].label,self.parse_tree[j].x,self.parse_tree[j].y,self.parse_tree[j].w,self.parse_tree[j].h)
			print("_______________________")
	def meta_training(self):

		# For all epochs
		for e in range(self.num_epochs):
			# For all images
			# for i in range(self.num_images):
			for i in range(1):
				print("Epoch:",e,"Training Image:",i)
				# Intialize the parse tree for this image.
				image = self.images[i]

				# Remember the state is pixel label, then x origin, y origin, width, height.
				# self.state = [0,0,0,self.image_size,self.image_size] 
				self.state = parse_tree_node(label=0,x=0,y=0,w=self.image_size,h=self.image_size)
				self.initialize_tree()

				print("Constructing Parse Tree.")
				self.construct_parse_tree(i)
				# WHEN THE PARSE IS COMPLETE, 
				# First just execute the set of trajectories in parse tree, by traversing the LEAF NODES in the order they appear in the tree (DFS-LR)
				# REsolve goals into global frame.
						
				#compute rewards for the chosen actions., then propagate them through the tree.
				print("Computing Rewards.")
				self.compute_rewards(i)
				print("Propagating Rewards.")
				self.propagate_rewards()
				print("Backprop.")
				self.backprop(i)


	############################
	# Pixel labels: 
	# 0 for shape
	# 1 for shape with primitive 1
	# 2 for shape with primitive 2
	# 3 for shape with primitive 3
	# 4 for shape with primitive 4
	# 5 for region with no primitive (not to be painted)
	############################

	def map_rules_to_indices(self, rule_index):
		if (rule_index==0)or(rule_index==1):
			return [0,0]
		if (rule_index==2)or(rule_index==3):
			return [1,0]
		if (rule_index==4)or(rule_index==5):
			return [0,1]
		if (rule_index==6):
			return 1
		if (rule_index==7):
			return 2

	############################
	# Rule numbers:
	# 0 (Shape) -> (Shape) (Shape) 								(Vertical split)
	# 1 (Shape) -> (Shape (Shape) 								(Horizontal split)
	# 2 (Shape) -> (Region with primitive #) (Shape)			(Vertical split)
	# 3 (Shape) -> (Region with primitive #) (Shape)			(Horizontal split)
	# 4 (Shape) -> (Shape) (Region with primitive #) 			(Vertical split)
	# 5 (Shape) -> (Shape) (Region with primitive #) 			(Horizontal split)
	# 6 (Shape) -> (Region with primitive #) 
	# 7 (Shape) -> (Region not to be painted)
	############################

def main(args):

	# # Create a TensorFlow session with limits on GPU usage.
	# gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list="1,2")
	# config = tf.ConfigProto(gpu_options=gpu_ops)
	# sess = tf.Session(config=config)

	# If CPU:
	sess = tf.Session()

	hierarchical_model = hierarchical()
	hierarchical_model.initialize_tensorflow_model(sess)

	# MUST LOAD IMAGES / LOAD NOISY IMAGES (So that the CNN has some features to latch on to.)	
	hierarchical_model.images = npy.load(str(sys.argv[1]))	
	hierarchical_model.true_labels = npy.load(str(sys.argv[2]))

	# CALL TRAINING
	hierarchical_model.meta_training()

if __name__ == '__main__':
	main(sys.argv)

