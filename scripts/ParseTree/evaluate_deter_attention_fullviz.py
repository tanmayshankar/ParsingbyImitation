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
		self.reward = 0.

	def disp(self):
		print("Label:", self.label)
		print("X:",self.x,"Y:",self.y,"W:",self.w,"H:",self.h)

		print("Backward Index:",self.backward_index)
		print("Reward:",self.reward)
		print("Rule:",self.rule_applied,"Split:",self.split)
		print("____________________________________________")

class hierarchical():

	def __init__(self):

		self.num_epochs = 1
		self.num_images = 5000
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
		# Now not using the start and goal
		self.fcs1_l1_shape = 120
		self.W_fcs1_l1 = tf.Variable(tf.truncated_normal([self.fc_input_shape,self.fcs1_l1_shape],stddev=0.1),name='W_fcs1_l1')
		self.b_fcs1_l1 = tf.Variable(tf.constant(0.1,shape=[self.fcs1_l1_shape]),name='b_fcs1_l1')
		self.fcs1_l1 = tf.nn.relu(tf.add(tf.matmul(self.relu_conv5_flat,self.W_fcs1_l1),self.b_fcs1_l1),name='fcs1_l1')

		# 2nd FC layer: RULE Output:
		self.number_primitives = 1
		# Now we have shifted to the 4 rule version of this: 
		# Horizontal split rule into two shapes
		# Vertical split rule into two shapes
		# Assignment rule to region with primitive.
		# Assignment rule to region without primitive.

		self.fcs1_output_shape = 1*self.number_primitives+5
		self.W_fcs1_l2 = tf.Variable(tf.truncated_normal([self.fcs1_l1_shape,self.fcs1_output_shape],stddev=0.1),name='W_fcs1_l2')
		self.b_fcs1_l2 = tf.Variable(tf.constant(0.1,shape=[self.fcs1_output_shape]),name='b_fcs1_l2')
		self.fcs1_presoftmax = tf.add(tf.matmul(self.fcs1_l1,self.W_fcs1_l2),self.b_fcs1_l2,name='fcs1_presoftmax')
		self.rule_probabilities = tf.nn.softmax(self.fcs1_presoftmax,name='softmax')
		
		# Defining training ops. 
		self.rule_return_weight = tf.placeholder(tf.float32,shape=(None),name='rule_return_weight')
		# self.split_return_weight = tf.placeholder(tf.float32,shape=(None),name='split_return_weight')
		self.target_rule = tf.placeholder(tf.float32,shape=( self.fcs1_output_shape),name='target_rule')

		# Defining the loss for each of the 3 streams, rule, split and goal.
		# Rule loss is the negative cross entropy between the rule probabilities and the chosen rule as a one-hot encoded vector. 
		# Weighted by the return obtained. This is just the negative log probability of the selected action.

		# NO NEGATIVE SIGN HERE - 13/6
		self.rule_loss = tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=self.target_rule,logits=self.fcs1_presoftmax),self.rule_return_weight)

		# Creating a training operation to minimize the total loss.
		self.train = tf.train.AdamOptimizer(1e-4).minimize(self.rule_loss,name='Adam_Optimizer')

		# Writing graph and other summaries in tensorflow.
		self.writer = tf.summary.FileWriter('training',self.sess.graph)
		# Creating a saver object to save models.
		self.saver = tf.train.Saver(max_to_keep=None)

		if model_file:
			print("RESTORING MODEL:")
			self.saver.restore(self.sess,model_file)
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)

	def save_model(self, model_index):
		save_path = self.saver.save(self.sess,'saved_models/model_{0}.ckpt'.format(model_index))

	# def load_model(self, model_file):
	# 	self.saver.restore(self.sess, model_file)
	# 	self.is_restored = True

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

		# rule_probabilities/=rule_probabilities.sum()
		selected_rule = npy.argmax(rule_probabilities)
		# selected_rule = npy.random.choice(range(self.fcs1_output_shape),p=rule_probabilities[0])
		indices = self.map_rules_to_indices(selected_rule)

		# If it is a split rule:
		if selected_rule<=3:
			# Apply the rule: if the rule number is even, it is a vertical split and if the current non-terminal to be parsed is taller than 1 unit:
			if ((selected_rule==0) or (selected_rule==2)):
				split_location = int(float(self.state.h)/2)
				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=self.state.w,h=split_location,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x,y=self.state.y+split_location,w=self.state.w,h=self.state.h-split_location,backward_index=self.current_parsing_index)

			if ((selected_rule==1) or (selected_rule==3)):
				split_location = int(float(self.state.w)/2)
				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=split_location,h=self.state.h,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x+split_location,y=self.state.y,w=self.state.w-split_location,h=self.state.h,backward_index=self.current_parsing_index)
				
			# Update current parse tree with split location and rule applied.
			self.parse_tree[self.current_parsing_index].split=split_location
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
			# for x in range(int(self.state.x),int(self.state.x+self.state.w)):
			# 	for y in range(int(self.state.y),int(self.state.y+self.state.h)):
			# 		self.painted_image[x,y] = 1

			# CHANGING PAINTING CONSTANT TO 2
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

			# If it is a region with a primitive.
			# if self.parse_tree[j].label==1:
			if self.parse_tree[j].label==1 or self.parse_tree[j].label==2:
				self.terminal_reward_nostartgoal(image_index)

			self.parse_tree[j].reward = copy.deepcopy(self.state.reward)

	def backprop(self, image_index):
		# Must decide whether to do this stochastically or in batches. # For now, do it stochastically, moving forwards through the tree.

		# NOW CHANGING TO 4 RULE SYSTEM.
		target_rule = npy.zeros(self.fcs1_output_shape)
		for j in range(len(self.parse_tree)):
			self.state = self.parse_tree[j]
			
			boundary_width = 2
			lowerx = max(0,self.state.x-boundary_width)
			upperx = min(self.image_size,self.state.x+self.state.w+boundary_width)
			lowery = max(0,self.state.y-boundary_width)
			uppery = min(self.image_size,self.state.y+self.state.h+boundary_width)

			attention_factor = 0.5
			self.attention = attention_factor*npy.ones((self.image_size,self.image_size))
			# self.attention[lowerx:upperx,lowery:uppery] = 1.
			self.attention[self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h] = 1
			# self.image_input = self.images[image_index]*self.attention


			# Pick up correct portion of image.
			# self.image_input = self.images[image_index, self.state.x:self.state.x+self.state.w, self.state.y:self.state.y+self.state.h]

			self.image_input = self.images[image_index,lowerx:upperx,lowery:uppery]
			self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))

			rule_weight = 0
			split_weight = 0
			target_rule = npy.zeros(self.fcs1_output_shape)

			# MUST PARSE EVERY NODE
			# If shape:
			if self.parse_tree[j].label==0:
				# If split rule.
				# if self.parse_tree[j].rule_applied<=5:
				# if self.parse_tree[j].rule_applied<=1:
				# 	# split_weight = self.parse_tree[j].reward
				# 	rule_weight = self.parse_tree[j].reward
				# 	target_rule[self.parse_tree[j].rule_applied] = 1.
				# # If rule 2 or rule 3.
				# if self.parse_tree[j].rule_applied>=2:
				# 	rule_weight = self.parse_tree[j].reward

				rule_weight = self.parse_tree[j].reward
				target_rule[self.parse_tree[j].rule_applied] = 1.

			# Here ,we only backprop for shapes, since we only choose actions for shapese.
				rule_loss, _ = self.sess.run([self.rule_loss, self.train], \
					feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1), \
						 self.rule_return_weight: rule_weight, self.target_rule: target_rule})

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


			attention_factor = 0.2
			self.attention = attention_factor*npy.ones((self.image_size,self.image_size))
			# self.attention[lowerx:upperx,lowery:uppery] = 1.
			self.attention[self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h] = 1
			# self.image_input = self.images[image_index]*self.attention
			self.image_input = self.images[image_index,lowerx:upperx,lowery:uppery]
			self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))

			# If the current non-terminal is a shape.
			if (self.state.label==0):
				# print("PARSING NON TERMINAL")
				self.parse_nonterminal(image_index)

			# If the current non-terminal is a region assigned a particular primitive.
			if (self.state.label==1):
				# print("________  PARSING TERMINAL")
				self.parse_primitive_terminal()
			
			if (self.state.label==2):
				self.current_parsing_index+=1

			# if (self.predicted_labels[image_index]==1).any():
			self.alternate_painted_image[npy.where(self.predicted_labels[image_index]==1)]=1.
			
			self.alternate_predicted_labels[npy.where(self.predicted_labels[image_index]==1)]=2.
			self.alternate_predicted_labels[npy.where(self.predicted_labels[image_index]==2)]=1.

			if self.plot:
				self.fig.suptitle("Processing Image: {0}".format(image_index))
				self.sc1.set_data(self.alternate_predicted_labels)
				# self.sc1.set_data(self.predicted_labels[image_index])
				self.sc2.set_data(self.true_labels[image_index])
				# self.sc3.set_data(self.painted_image)
				self.sc3.set_data(self.alternate_painted_image)
				self.sc4.set_data(self.images[image_index])

				self.sc5.set_data(self.attention)
				self.sc6.set_data(self.image_input)
				self.fig.canvas.draw()
				plt.pause(0.001)

		# for j in range(len(self.parse_tree)):
		# 	self.parse_tree[j].disp()

	def meta_training(self, train=True):

		image_index = 0
		self.painted_image = -npy.ones((self.image_size,self.image_size))
		self.attention = npy.ones((self.image_size,self.image_size))

		if self.plot:
			self.fig, self.ax = plt.subplots(1,6,sharey=True)
			# plt.ion()
			# plt.show()
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

			self.sc5 = self.ax[4].imshow(self.attention,aspect='equal')
			self.sc5.set_clim([-1,1])
			self.ax[4].set_title("Attention")
			self.ax[4].set_adjustable('box-forced')

			self.sc6 = self.ax[5].imshow(self.attention,aspect='equal')
			self.sc6.set_clim([-1,1])
			self.ax[5].set_title("Attended Image")
			self.ax[5].set_adjustable('box-forced')

			self.fig.canvas.draw()
			plt.pause(0.001)

		# For all epochs
		if not(train):
			self.num_epochs=1
			
		for e in range(self.num_epochs):	
			# For all images
			for i in range(self.num_images):	
			# for i in range(1):		
				
				print("#________________________________________________________________#")
				print("Epoch:",e,"Training Image:",i)
				print("#________________________________________________________________#")

				# Intialize the parse tree for this image.=
				self.state = parse_tree_node(label=0,x=0,y=0,w=self.image_size,h=self.image_size)
				self.initialize_tree()

				self.construct_parse_tree(i)	
				#compute rewards for the chosen actions., then propagate them through the tree.
				self.compute_rewards(i)
				self.propagate_rewards()
				print("Parsing Image:",i)
				
				print("TOTAL REWARD:",self.parse_tree[0].reward)
				if train:
					self.backprop(i)

			if train:
				npy.save("halfparsed_clean3_{0}.npy".format(e),self.predicted_labels)
			else:
				npy.save("validation_3.npy".format(e),self.predicted_labels)

			self.evaluate()
			self.predicted_labels = npy.zeros((20000,20,20))

			# self.save_model(e)

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

	def evaluate(self):

		pred_label = copy.deepcopy(self.predicted_labels)
		pred_label[npy.where(pred_label==2)]=-1

		print("The image correlation:")
		print((self.true_labels*pred_label).sum()/((self.image_size**2)*self.num_images))
		print("Wrong pixel fraction:")
		print(abs(self.true_labels-pred_label).sum()/(2*(self.image_size**2)*self.num_images))

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
	hierarchical_model.plot = 1
	
	if sys.argv[3]:
		print("HI!")
		model_file = str(sys.argv[3])
		hierarchical_model.initialize_tensorflow_model(sess,model_file)
	else:
		hierarchical_model.initialize_tensorflow_model(sess)

	# CALL TRAINING
	hierarchical_model.meta_training(train=False)

if __name__ == '__main__':
	main(sys.argv)



