#!/usr/bin/env python
from headers import *
from state_class import *

class ModularNet():

	def __init__(self):

		self.num_epochs = 20
		self.save_every = 2
		self.num_images = 163
		self.current_parsing_index = 0
		self.parse_tree = [parse_tree_node()]
		self.paintwidth = -1
		self.minimum_width = -1
		self.images = []
		self.original_images = []
		self.true_labels = []
		self.image_size = 256
		self.intermittent_lambda = 0.
		self.suffix = []

		# For Epsilon Greedy Policy: 
		self.initial_epsilon = 0.9
		self.final_epsilon = 0.1
		self.decay_epochs = 5
		self.annealing_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_epochs*self.num_images)
		self.annealed_epsilon = 0.

		# Maintaining list of all goals and start locations. 
		self.goal_list = []
		self.start_list = []
		self.previous_goal = npy.zeros(2)
		self.current_start = npy.zeros(2)

		self.max_parse_steps = 9
		#If the parse tree length goes greater than this, it assigns all unparsed non-terminals to terminals. 	

	def load_base_model(self, sess, model_file=None):

		# Define a KERAS / tensorflow session. 
		self.sess = sess
		# We are always loading the model from the the gradient file at the least. 
		self.base_model = keras.models.load_model(model_file)
		
		# The common FC layers are defined here: Input features are keras layer outputs of base model. 
		for layers in self.base_model.layers:
			if layers.name=='fc6_features':
				self.fc6_features = layers.output
			if layers.name=='vertical_grads':
				self.vertical_split_probs = layers.output
				# self.vertical_split_probs = layers
			if layers.name=='horizontal_grads':
				self.horizontal_split_probs = layers.output
				# self.horizontal_split_probs = layers

	def define_rule_stream(self):
		# Now defining rule FC:
		self.rule_num_branches = 4
		self.target_rule_shapes = [6,4,4,2]
		self.rule_num_fclayers = 2
		self.rule_num_hidden = 256

		self.rule_fc = [keras.layers.Dense(self.rule_num_hidden,activation='relu')(self.fc6_features) for j in range(self.rule_num_branches)]
		self.rule_probabilities = [keras.layers.Dense(self.target_rule_shapes[j],activation='softmax',name='rule_probabilities{0}'.format(j))(self.rule_fc[j]) for j in range(self.rule_num_branches)]
		# self.rule_loss_weight = [keras.backend.variable(npy.zeros(1),dtype='float64',name='rule_loss_weight{0}'.format(j)) for j in range(self.rule_num_branches)]
		self.rule_loss_weight = [keras.backend.variable(0.,name='rule_loss_weight{0}'.format(j)) for j in range(self.rule_num_branches)]

	def define_split_stream(self):
		# self.split_indicator = keras.layers.Input(batch_shape=(1,),dtype='int32',name='split_indicator')
		self.split_loss_weight = [keras.backend.variable(0.,name='split_loss_weight{0}'.format(j)) for j in range(2)]
		# self.split_loss_weight = [keras.backend.variable(npy.zeros(1),dtype='float64',name='split_loss_weight{0}'.format(j)) for j in range(2)]

	def define_primitive_stream(self):
		# Defining primitive FC layers.
		self.primitive_num_hidden = 256
		self.num_primitives = 4

		self.primitive_fc0 = keras.layers.Dense(self.primitive_num_hidden,activation='relu')(self.fc6_features)
		self.primitive_probabilities = keras.layers.Dense(self.num_primitives,activation='softmax',name='primitive_probabilities')(self.primitive_fc0)		

		self.primitive_targets = keras.backend.placeholder(shape=(self.num_primitives),name='primitive_targets')
		# self.primitive_loss_weight = keras.backend.variable(npy.zeros(1),dtype='float64',name='primitive_loss_weight')
		self.primitive_loss_weight = keras.backend.variable(0.,name='primitive_loss_weight')

	def define_keras_model(self):
		############################################################################################### 
	
		# Defining the new model.
		self.model = keras.models.Model(inputs=self.base_model.input,
										outputs=[self.rule_probabilities[0],self.rule_probabilities[1],self.rule_probabilities[2],self.rule_probabilities[3],
												 self.horizontal_split_probs,self.vertical_split_probs,self.primitive_probabilities])
		print("Model successfully defined.")
			
		# Defining optimizer.
		self.adam_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

		# # Option to freeze base model layers.
		# for layer in self.base_model.layers:
		# 	layer.trainable = False
		
		# Compiling the new model
		# Now not feeding separate losses or target tensors. Just setting loss weights as Keras variables. 
		self.model.compile(optimizer=self.adam_optimizer,loss='categorical_crossentropy',loss_weights={'rule_probabilities0': self.rule_loss_weight[0],
																									   'rule_probabilities1': self.rule_loss_weight[1],
																									   'rule_probabilities2': self.rule_loss_weight[2],
																									   'rule_probabilities3': self.rule_loss_weight[3],
																									   'horizontal_grads': self.split_loss_weight[0],
																									   'vertical_grads': self.split_loss_weight[1],
																									   'primitive_probabilities': self.primitive_loss_weight})	
		print("Supposed to have compiled model.")
		# embed()
		# Because of how this is defined, we must call a Callback in Fit. 

	def save_model_weights(self,k):
		self.model.save_weights("model_weights_epoch{0}.h5".format(k))

	def save_model(self,k):
		self.model.save("model_file_epoch{0}.h5".format(k))

	def load_pretrained_model(self, model_file):
		# Load the model - instead of defining from scratch.
		self.model = keras.models.load_model(model_file)

		# with open(model_file,"r") as f:
		# 	self.model = keras.models.model_from_yaml(f.read())

		# # Load the weights:
		# self.model.load_weights(weight_file)
		# adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		# self.model.compile(optimizer=adam,loss='categorical_crossentropy')

	def load_model_weights(self,weight_file):
		self.model.load_weights(weight_file)

	def create_modular_net(self, sess, load_pretrained_mod=False, model_file=None):
		if load_pretrained_mod:
			self.load_pretrained_model(model_file)
		else:	
			print("Training Policy from base model.")
			self.load_base_model(sess, model_file)
			self.define_rule_stream()
			self.define_split_stream()
			self.define_primitive_stream()
			self.define_keras_model()

	###########################################################################################
	############################## NOW MOVING TO PARSING CODE #################################
	###########################################################################################

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

	# Checked this - should be good - 11/1/18
	def parse_nonterminal(self, image_index, max_parse=False):

		if max_parse:
			self.state.rule_indicator = 3
		else:
			# Four branches of the rule policy.
			self.set_rule_indicator()

		# rule_probabilities = self.sess.run(self.selected_rule_probabilities, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,1),  self.rule_indicator: self.state.rule_indicator})

		# self.model.predict(self.resized_image.reshape(1,self.image_size,self.image_size,3))
		# self.sess.run(self.model.output[-1],feed_dict={self.model.input: self.resized_image.reshape(1,self.image_size,self.image_size,3)})
		# rule_probabilities = self.sess.run(self.rule_probabilities[self.state.rule_indicator],feed_dict={self.model.input: self.resized_image.reshape(1,self.image_size,self.image_size,3)})
			
		rule_probabilities = self.model.predict(self.resized_image.reshape(1,self.image_size,self.image_size,3))[self.state.rule_indicator]
		epsgreedy_rule_probs = npy.ones((rule_probabilities.shape[-1]))*(self.annealed_epsilon/rule_probabilities.shape[-1])
		epsgreedy_rule_probs[rule_probabilities.argmax()] = 1.-self.annealed_epsilon+self.annealed_epsilon/rule_probabilities.shape[-1]

		# Must handle the fact that branches now index rules differently, using remap_rule_indices.
		# CHECK IF ITS rule_probabilities[0] still
		if self.to_train:
			selected_rule = npy.random.choice(range(len(rule_probabilities[0])),p=epsgreedy_rule_probs)
		elif not(self.to_train):
			selected_rule = npy.argmax(rule_probabilities[0])

		self.parse_tree[self.current_parsing_index].rule_applied = copy.deepcopy(selected_rule)
		selected_rule = self.remap_rule_indices(selected_rule)
		indices = self.map_rules_to_state_labels(selected_rule)
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

					# split_probs = self.sess.run(self.horizontal_split_probs, feed_dict={self.model.input: self.resized_image.reshape(1,self.image_size,self.image_size,3)})
					split_probs = self.model.predict(self.resized_image.reshape(1,self.image_size,self.image_size,3))[4]

					epsgreedy_split_probs = npy.ones((self.image_size))*(self.annealed_epsilon/self.image_size)						
					epsgreedy_split_probs[split_probs.argmax()] = 1.-self.annealed_epsilon+self.annealed_epsilon/self.image_size
					# embed()
					counter+=1

					split_location = npy.random.choice(range(self.image_size),p=epsgreedy_split_probs)
					inter_split = copy.deepcopy(split_location)

					if inter_split>(self.state.h/2):
						split_location = int(npy.floor(float(inter_split)*self.state.h/self.image_size))
					else:
						split_location = int(npy.ceil(float(inter_split)*self.state.h/self.image_size))

					if counter>25:
						print("State: H",self.state.h, "Split fraction:",inter_split, "Split location:",split_location)
			
				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=self.state.w,h=split_location,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x,y=self.state.y+split_location,w=self.state.w,h=self.state.h-split_location,backward_index=self.current_parsing_index)

			if ((selected_rule==1) or (selected_rule==3)):
				counter = 0
				self.state.split_indicator = 1

				# SAMPLING SPLIT LOCATION INSIDE THIS CONDITION:
				while (split_location<=0)or(split_location>=self.state.w):

					# split_probs = self.sess.run(self.vertical_split_probs, feed_dict={self.model.input: self.resized_image.reshape(1,self.image_size,self.image_size,3)})
					split_probs = self.model.predict(self.resized_image.reshape(1,self.image_size,self.image_size,3))[5]
					epsgreedy_split_probs = npy.ones((self.image_size))*(self.annealed_epsilon/self.image_size)						
					epsgreedy_split_probs[split_probs.argmax()] = 1.-self.annealed_epsilon+self.annealed_epsilon/self.image_size

					counter+=1

					split_location = npy.random.choice(range(self.image_size),p=epsgreedy_split_probs)
					inter_split = copy.deepcopy(split_location)
	
					if inter_split>(self.state.w/2):
						split_location = int(npy.floor(float(inter_split)*self.state.w/self.image_size))
					else:
						split_location = int(npy.ceil(float(inter_split)*self.state.w/self.image_size))

					if counter>25:
						print("State: W",self.state.w, "Split fraction:",inter_split, "Split location:",split_location)

				# Create splits.
				s1 = parse_tree_node(label=indices[0],x=self.state.x,y=self.state.y,w=split_location,h=self.state.h,backward_index=self.current_parsing_index)
				s2 = parse_tree_node(label=indices[1],x=self.state.x+split_location,y=self.state.y,w=self.state.w-split_location,h=self.state.h,backward_index=self.current_parsing_index)
				
			# Update current parse tree with split location and rule applied.
			self.parse_tree[self.current_parsing_index].split = inter_split
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

	# Checked this - should be good - 11/1/18
	def parse_primitive_terminal(self, image_index):
		# Sample a goal location.
		self.nonpaint_moving_term = 0.
		self.strokelength_term = 0.

		# If it is a region to be painted and assigned a primitive:
		if (self.state.label==1):
			# primitive_probabilities = self.sess.run(self.primitive_probabilities, feed_dict={self.input: self.resized_image.reshape(1,self.image_size,self.image_size,3)})				
			primitive_probabilities = self.model.predict(self.resized_image.reshape(1,self.image_size,self.image_size,3))[-1]
			epsgreedy_primitive_probs = npy.ones((self.num_primitives))*(self.annealed_epsilon/self.num_primitives)
			epsgreedy_primitive_probs[primitive_probabilities.argmax()] = 1.-self.annealed_epsilon +self.annealed_epsilon/self.num_primitives

			if self.to_train:
				# selected_primitive = npy.random.choice(range(self.num_primitives),p=primitive_probabilities[0])
				selected_primitive = npy.random.choice(range(self.num_primitives),p=epsgreedy_primitive_probs)
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
		
		self.alpha = 1.1
		
		# Non-linearizing rewards.
		for j in range(len(self.parse_tree)):
			self.parse_tree[j].reward = npy.tan(self.alpha*self.parse_tree[j].reward)		

			# # Additional term for continuity. 
		for j in range(len(self.parse_tree)):
			self.parse_tree[j].reward += self.parse_tree[j].intermittent_term*self.intermittent_lambda
			# print("MODIFIED REWARD:",self.parse_tree[j].reward)

	# Checked this - should be good - 11/1/18
	def backprop(self, image_index):
		# Must decide whether to do this stochastically or in batches. # For now, do it stochastically, moving forwards through the tree.
		for j in range(len(self.parse_tree)):
			
			self.state = self.parse_tree[j]
			boundary_width = 0
			lowerx = max(0,self.state.x-boundary_width)
			upperx = min(self.image_size,self.state.x+self.state.w+boundary_width)
			lowery = max(0,self.state.y-boundary_width)
			uppery = min(self.image_size,self.state.y+self.state.h+boundary_width)

			self.image_input = self.images[image_index, lowerx:upperx, lowery:uppery]
			self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))

			# What we really need to set is the loss weights and targets. Then you can just call Keras fit. 

			# Declare target rule and primitives.
			target_rule = [npy.zeros(self.target_rule_shapes[k]) for k in range(self.rule_num_branches)]
			target_splits = [npy.zeros(self.image_size) for k in range(2)]
			target_primitive = npy.zeros(self.num_primitives)

			# # Declare target rule and primitives.
			# target_rule = [npy.ones(self.target_rule_shapes[k])/self.target_rule_shapes[k] for k in range(self.rule_num_branches)]
			# target_splits = [npy.ones(self.image_size)/self.image_size for k in range(2)]
			# target_primitive = npy.ones(self.num_primitives)/self.num_primitives

			# Set the return weight for the loss globally., i.e. for all losses.
			# return_weight = self.parse_tree[j].reward
			return_weight = self.parse_tree[j].reward

			# # Here, we set the indicator functions for the various cases.
			# # If it's a non-terminal:
			# if self.parse_tree[j].label==0:			
			# 	target_rule
			# 	# A rule was necessarily applied, so set the target rule and the rule branch indicator.
				

			# 	# target_rule[self.parse_tree[j].rule_applied] = 1.
			# 	# print("Ind:")
			# 	# print(self.parse_tree[j].rule_indicator)
			# 	# print(self.parse_tree[j].rule_applied)

			# 	target_rule[self.parse_tree[j].rule_indicator][self.parse_tree[j].rule_applied] = 1.
			# 	rule_indicator = self.parse_tree[j].rule_indicator
			# 	# If it was a split rule, then both the split and rule policies were used. 
			# 	if self.parse_tree[j].rule_applied<=3:
			# 		policy_indicator = 0
			# 		# Since a split location was sampled, provide the split indicator for branch.
			# 		split_indicator = self.parse_tree[j].split_indicator
			# 	else:
			# 		policy_indicator = 1					

			for k in range(self.rule_num_branches):
				keras.backend.set_value(self.rule_loss_weight[k],0.)
			for k in range(2):
				keras.backend.set_value(self.split_loss_weight[k],0.)

			keras.backend.set_value(self.primitive_loss_weight,0.)

			# If it was a non terminal:
			if self.parse_tree[j].label == 0:
				# target_rule[self.parse_tree[j].rule_indicator][:] = 0.
				target_rule[self.parse_tree[j].rule_indicator][self.parse_tree[j].rule_applied] = 1.
				keras.backend.set_value(self.rule_loss_weight[self.parse_tree[j].rule_indicator],return_weight)

				if self.parse_tree[j].rule_applied<=3:					
					keras.backend.set_value(self.split_loss_weight[self.parse_tree[j].rule_applied%2],return_weight)
					
					if self.parse_tree[j].rule_applied%2==0:
						target_splits[0][self.parse_tree[j].split] = 1.
					if self.parse_tree[j].rule_applied%2==1:
						target_splits[1][self.parse_tree[j].split] = 1.
				
			# If it was a terminal symbol that was to be painted:
			if self.parse_tree[j].label==1:
				# Set the target primitive and policy branch.
				# target_primitive[:] = 0.
				target_primitive[self.parse_tree[j].primitive] = 1.				
				keras.backend.set_value(self.primitive_loss_weight,return_weight)

			# # Remember, we don't backprop for a terminal not to be painted (since we already would've backpropagated gradients
			# # for assigning the parent non-terminal to a region not to be painted).

			self.model.fit(x=self.resized_image.reshape((1,self.image_size,self.image_size,3)),y={'rule_probabilities0': target_rule[0].reshape((1,self.target_rule_shapes[0])),
																								  'rule_probabilities1': target_rule[1].reshape((1,self.target_rule_shapes[1])),
																								  'rule_probabilities2': target_rule[2].reshape((1,self.target_rule_shapes[2])),
																								  'rule_probabilities3': target_rule[3].reshape((1,self.target_rule_shapes[3])),
																								  'horizontal_grads': target_splits[0].reshape((1,self.image_size)),
																								  'vertical_grads': target_splits[1].reshape((1,self.image_size)),
																								  'primitive_probabilities': target_primitive.reshape((1,self.num_primitives))})	

	# Checked this - should be good - 11/1/18
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

			self.image_input = self.images[image_index, lowerx:upperx, lowery:uppery]

			self.imagex = upperx-lowerx
			self.imagey = uppery-lowery
			self.ux = upperx
			self.uy = uppery
			self.lx = lowerx
			self.ly = lowery

			self.resized_image = cv2.resize(self.image_input,(self.image_size,self.image_size))

			# If we exceeded the number of parse steps allowed, AND there are non-terminals left, assign them to TERMINAL symbols. 		
			if (len(self.parse_tree) > self.max_parse_steps):
				# If the current non-terminal is a shape.
				if (self.state.label==0):
					self.parse_nonterminal(image_index, max_parse=True)
				# If the current non-terminal is a region assigned a particular primitive.
				if (self.state.label==1) or (self.state.label==2):
					self.parse_primitive_terminal(image_index)

			else:
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
	
		# self.alternate_painted_image[npy.where(self.predicted_labels[image_index]==1)]=1.
		self.alternate_painted_image[npy.where(self.painted_images[image_index]==1)]=1.
		self.alternate_predicted_labels[npy.where(self.predicted_labels[image_index]==1)]=1.
		self.alternate_predicted_labels[npy.where(self.predicted_labels[image_index]==2)]=-1.

		if self.plot:
			self.fig.suptitle("Processing Image: {0}".format(image_index)) 
			self.sc1.set_data(self.alternate_predicted_labels)
			self.attention_plots()
			self.sc2.set_data(self.mask)
			self.sc3.set_data(self.images[image_index])
			self.sc4.set_data(self.alternate_painted_image)

			# Plotting split line segments from the parse tree.
			split_segs = []
			for j in range(len(self.parse_tree)):

				colors = ['r']

				if self.parse_tree[j].label==0:

					rule_app_map = self.remap_rule_indices(self.parse_tree[j].rule_applied)

					if (self.parse_tree[j].alter_rule_applied==1) or (self.parse_tree[j].alter_rule_applied==3):
						sc = self.parse_tree[j].boundaryscaled_split
						split_segs.append([[self.parse_tree[j].y,self.parse_tree[j].x+sc],[self.parse_tree[j].y+self.parse_tree[j].h,self.parse_tree[j].x+sc]])
						
					if (self.parse_tree[j].alter_rule_applied==0) or (self.parse_tree[j].alter_rule_applied==2):					
						sc = self.parse_tree[j].boundaryscaled_split
						split_segs.append([[self.parse_tree[j].y+sc,self.parse_tree[j].x],[self.parse_tree[j].y+sc,self.parse_tree[j].x+self.parse_tree[j].w]])

			split_lines = LineCollection(split_segs, colors='k', linewidths=2)
			split_lines2 = LineCollection(split_segs, colors='k',linewidths=2)
			split_lines3 = LineCollection(split_segs, colors='k',linewidths=2)

			self.split_lines = self.ax[1].add_collection(split_lines)				
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
				self.lines = self.ax[0].add_collection(lines)	
			
			self.fig.canvas.draw()
			# raw_input("Press any key to continue.")
			plt.pause(0.1)	
			# plt.pause(0.5)	

			if len(self.ax[0].collections):
				del self.ax[0].collections[-1]
		
			del self.ax[3].collections[-1]
			del self.ax[2].collections[-1]			
			del self.ax[1].collections[-1]

	def define_plots(self):
		image_index = 0
		if self.plot:
			self.fig, self.ax = plt.subplots(1,4,sharey=True)
			self.fig.show()
			
			self.sc1 = self.ax[0].imshow(self.predicted_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			# self.sc1 = self.ax[0].imshow(self.predicted_labels[image_index],aspect='equal',cmap='jet')
			self.sc1.set_clim([-1,1])
			# self.sc1.set_clim([0,2])
			self.ax[0].set_title("Predicted Labels")
			self.ax[0].set_adjustable('box-forced')
			# self.ax[0].set_xlim(self.ax[0].get_xlim()[0]-0.5, self.ax[0].get_xlim()[1]+0.5) 
			# self.ax[0].set_ylim(self.ax[0].get_ylim()[0]-0.5, self.ax[0].get_ylim()[1]+0.5) 

			self.sc2 = self.ax[1].imshow(self.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			# self.sc2 = self.ax[1].imshow(self.true_labels[image_index],aspect='equal',cmap='jet')
			self.sc2.set_clim([-1,1])
			self.ax[1].set_title("Parse Tree")
			self.ax[1].set_adjustable('box-forced')

			self.sc3 = self.ax[2].imshow(self.images[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			# self.sc3 = self.ax[2].imshow(self.images[image_index],aspect='equal',cmap='jet')
			self.sc3.set_clim([-1,1.2])
			self.ax[2].set_title("Actual Image")
			self.ax[2].set_adjustable('box-forced')

			self.sc4 = self.ax[3].imshow(self.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			# self.sc4 = self.ax[3].imshow(self.true_labels[image_index],aspect='equal',cmap='jet') #, extent=[0,self.image_size,0,self.image_size],origin='lower')
			self.sc4.set_clim([-1,1])
			self.ax[3].set_title("Segmented Painted Image")
			self.ax[3].set_adjustable('box-forced')			

			self.fig.canvas.draw()
			plt.pause(0.1)	
	
	# Checked this - should be good - 11/1/18
	def meta_training(self,train=True):
		
		image_index = 0
		self.painted_image = -npy.ones((self.image_size,self.image_size))
		self.predicted_labels = npy.zeros((self.num_images,self.image_size, self.image_size))
		self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))
		# self.minimum_width = self.paintwidth
		
		if self.plot:			
			self.define_plots()

		self.to_train = train		
		if not(train):
			self.num_epochs=1

		# For all epochs
		for e in range(self.num_epochs):
			self.save_model_weights(e)				
			for i in range(self.num_images):

				self.initialize_tree()
				self.construct_parse_tree(i)
				self.propagate_rewards()				

				print("#___________________________________________________________________________")
				print("Epoch:",e,"Training Image:",i,"TOTAL REWARD:",self.parse_tree[0].reward)
	
				if train:
					self.backprop(i)
					if e<self.decay_epochs:
						epsilon_index = e*self.num_images+i
						self.annealed_epsilon = self.initial_epsilon-epsilon_index*self.annealing_rate
					else: 
						self.annealed_epsilon = self.final_epsilon

				self.start_list = []
				self.goal_list = []
								
			if train:
				npy.save("parsed_{0}.npy".format(e),self.predicted_labels)
				npy.save("painted_images_{0}.npy".format(e),self.painted_images)

				if ((e%self.save_every)==0):
					self.save_model_weights(e)				
			else: 
				npy.save("validation_{0}.npy".format(self.suffix),self.predicted_labels)
				npy.save("validation_painted_{0}.npy".format(self.suffix))
				
			self.predicted_labels = npy.zeros((self.num_images,self.image_size,self.image_size))
			self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))

	def map_rules_to_state_labels(self, rule_index):
		if (rule_index<=3):
			return [0,0]
		if (rule_index==4):
			return 1
		if (rule_index==5):
			return 2

	def preprocess(self):
		for i in range(self.num_images):
			self.images[i] = cv2.cvtColor(self.images[i],cv2.COLOR_RGB2BGR)
		
		self.images = self.images.astype(float)
		self.images -= self.images.mean(axis=(0,1,2))
		
		# self.image_gradients = npy.zeros((self.num_images,2,self.image_size[0]))

		# for i in range(self.num_images):
		# 	self.image_gradients[i,0,:-1] = self.gradients[i][0]
		# 	self.image_gradients[i,1,:-1] = self.gradients[i][1]

def parse_arguments():
	parser = argparse.ArgumentParser(description='Primitive-Aware Segmentation Argument Parsing')
	parser.add_argument('--images',dest='images',type=str)
	parser.add_argument('--labels',dest='labels',type=str)
	parser.add_argument('--size',dest='size',type=int)
	parser.add_argument('--paintwidth',dest='paintwidth',type=int)
	parser.add_argument('--minwidth',dest='minimum_width',type=int)
	parser.add_argument('--lambda',dest='inter_lambda',type=float)
	parser.add_argument('--base_model',dest='base_model',type=str)
	parser.add_argument('--suffix',dest='suffix',type=str)
	parser.add_argument('--gpu',dest='gpu')
	parser.add_argument('--plot',dest='plot',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	print(args)

	# # Create a TensorFlow session with limits on GPU usage.
	gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list=args.gpu)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	keras.backend.tensorflow_backend.set_session(sess)
	keras.backend.set_session(sess)
	keras.backend.set_learning_phase(1)

	# with sess.as_default():
	hierarchical_model = ModularNet()
	hierarchical_model.create_modular_net(sess,load_pretrained_mod=False,model_file=args.base_model)

	hierarchical_model.images = npy.load(args.images)
	hierarchical_model.true_labels = npy.load(args.labels) 
	hierarchical_model.image_size = args.size 
	hierarchical_model.preprocess()

	hierarchical_model.paintwidth = args.paintwidth
	hierarchical_model.minimum_width = args.minimum_width
	hierarchical_model.intermittent_lambda = args.inter_lambda

	hierarchical_model.plot = args.plot
	hierarchical_model.to_train = args.train
	
	if hierarchical_model.to_train:
		hierarchical_model.suffix = args.suffix
	
	hierarchical_model.meta_training(train=args.train)

if __name__ == '__main__':
	main(sys.argv)