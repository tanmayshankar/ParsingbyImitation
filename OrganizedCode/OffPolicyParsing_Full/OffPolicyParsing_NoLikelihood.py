#!/usr/bin/env python
from headers import *
from state_class import parse_tree_node
import EntropySplits

class Parser():

	# In this class, we are going to learn assignments and 
	def __init__(self, model_instance=None, data_loader_instance=None, memory_instance=None, args=None, session=None):

		self.model = model_instance
		self.data_loader = data_loader_instance
		self.memory = memory_instance
		self.args = args
		self.sess = session
		self.batch_size = 25
		self.num_epochs = 250
		self.save_every = 1
		self.max_parse_steps = 5
		self.minimum_width = 25

		# Parameters for annealing covariance. 
		self.initial_cov = 0.1
		self.final_cov = 0.01
		self.anneal_epochs = 80
		self.anneal_rate = (self.initial_cov-self.final_cov)/self.anneal_epochs

		self.initial_epsilon = 0.3
		self.final_epsilon = 0.05
		self.anneal_epsilon_rate = (self.initial_epsilon-self.final_epsilon)/self.anneal_epochs
		self.annealed_epsilon = copy.deepcopy(self.initial_epsilon)

	def initialize_tree(self,i):
		# Intialize the parse tree for this image.=
		self.state = parse_tree_node(label=0,x=0,y=0,w=self.data_loader.image_size,h=self.data_loader.image_size)
		self.state.image_index = copy.deepcopy(i)
		self.current_parsing_index = 0
		self.parse_tree = [parse_tree_node()]
		self.parse_tree[self.current_parsing_index]=self.state

	def append_parse_tree(self):
		for k in range(len(self.parse_tree)):
			# Only adding non-terminal states to the memory. 
			if self.parse_tree[k].label==0:
				self.memory.append_to_memory(self.parse_tree[k])

	def burn_in(self):
		
		# For one epoch, parse all images, store in memory.
		image_index_list = range(self.data_loader.num_images)
		npy.random.shuffle(image_index_list)

		for i in range(self.data_loader.num_images):
			print("Burning in image:",i)
			# Initialize tree.
			self.initialize_tree(image_index_list[i])

			self.set_parameters(0)

			# Parse Image.
			self.construct_parse_tree(image_index_list[i])

			# Compute rewards.
			# self.compute_rewards()
			self.backward_tree_propagation()

			# For every state in the parse tree, push to memory.
			self.append_parse_tree()

	def set_parameters(self,e):
		if self.args.train:
			if e<self.anneal_epochs:
				self.covariance_value = self.initial_cov - self.anneal_rate*e
			else:
				self.covariance_value = self.final_cov
			# print("Setting covariance as:",self.covariance_value)

			if e<self.anneal_epochs:
				self.annealed_epsilon = self.initial_epsilon-e*self.anneal_epsilon_rate
			else:
				self.annealed_epsilon = self.final_epsilon
		else:
			self.covariance_value = self.final_cov
			self.annealed_epsilon = self.final_epsilon
		
		# self.annealed_epsilon = self.final_epsilon

	def set_rule_mask(self):
		# Now we are going to allow 4 rules: 
		# (0) Split horizontally.
		# (1) Split vertically. 
		# (2) Assign to paint.
		# (3) Assign to not paint.

		# Earlier it was: 
		# Split horizontally
		# Assign to paint.
		# Assign to not paint.
		self.state.rule_mask = npy.zeros((self.model.num_rules))

		if len(self.parse_tree)>=self.max_parse_steps:
			# Allow only assignment.
			self.state.rule_mask[[2,3]] = 1.
		elif self.state.w<=self.minimum_width:
			self.state.rule_mask[[1,2,3]] = 1.
		elif self.state.h<=self.minimum_width:
			self.state.rule_mask[[0,2,3]] = 1.
		else:
			self.state.rule_mask[:] = 1.

	def select_rule_behavioural_policy(self):

		rule_probabilities = npy.ones((self.model.num_rules))*self.annealed_epsilon/self.model.num_rules

		entropy_image_input = self.data_loader.images[self.state.image_index,self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h]

		# Now returning both greedy axes and greedy splits. 
		self.greedy_axis,self.greedy_split = EntropySplits.bestsplit(entropy_image_input)
		
		# If there was no information gain maximizing (or entropy reducing) split, 
		# or if either of the split rules were banned. 
		if (self.greedy_axis==-1) or (self.greedy_split==-1) or ((self.state.rule_mask[:2]==0).all()):				
			
			# The sum value of the image is >0 for paint, <0 for non-paint.
			ip_img_sum = entropy_image_input.sum()

			if ip_img_sum>0:
				rule_probabilities[2] = 1.-self.annealed_epsilon+self.annealed_epsilon/self.model.num_rules
			elif ip_img_sum<0:
				rule_probabilities[3] = 1.-self.annealed_epsilon+self.annealed_epsilon/self.model.num_rules
			else:
				# The only reason this should have happened was because the 
				# vertical splits were banned, and the sum of ip_img_sum was 0. 

				print("I DON'T KNOW WHAT'S HAPPENING!")
				embed()
				rule_probabilities[[2,3]] = self.annealed_epsilon/self.model.num_rules

		else: 
			# Otherwise choose a split rule (eps greedy)
			# Now had to change this to self.greedy_axis. 
			rule_probabilities[self.greedy_axis] = 1.-self.annealed_epsilon+self.annealed_epsilon/self.model.num_rules

		masked_rule_probs = npy.multiply(self.state.rule_mask,rule_probabilities)
		masked_rule_probs/=masked_rule_probs.sum()

		self.state.rule_applied = npy.random.choice(range(3),p=masked_rule_probs)

	def insert_node(self, state, index):
		self.parse_tree.insert(index,state)

	def process_splits_behavioural_policy(self):

		# Earlier, were only taking care of horizontal splits. 
		# Modifying to general case. 

		split_probs = npy.zeros((self.data_loader.image_size-1))

		# If it's a horizontal rule:
		if self.state.rule_applied==0:
			split_probs[self.state.x+1:self.state.x+self.state.w] = self.annealed_epsilon/(self.state.w-1)
			self.greedy_split+=self.state.x
	
			if self.greedy_split>self.state.x and self.greedy_split<(self.state.x+self.state.w):
				split_probs[self.greedy_split] = 1.-self.annealed_epsilon+self.annealed_epsilon/(self.state.w-1)
	
			split_probs/=split_probs.sum()

			self.state.boundaryscaled_split = npy.random.choice(range(self.data_loader.image_size-1),p=split_probs)
			self.state.split = float(self.state.boundaryscaled_split-self.state.x)/self.state.w		
			# Transform to local patch coordinates.
			self.state.boundaryscaled_split -= self.state.x
			# Must add resultant states to parse tree.
			state1 = parse_tree_node(label=0,x=self.state.x,y=self.state.y,w=self.state.boundaryscaled_split,h=self.state.h,backward_index=self.current_parsing_index)
			state2 = parse_tree_node(label=0,x=self.state.x+self.state.boundaryscaled_split,y=self.state.y,w=self.state.w-self.state.boundaryscaled_split,h=self.state.h,backward_index=self.current_parsing_index)

		# If it's a vertical rule:
		if self.state.rule_applied==1:

			split_probs[self.state.y+1:self.state.y+self.state.h] = self.annealed_epsilon/(self.state.h-1)
			self.greedy_split+=self.state.y

			if self.greedy_split>self.state.y and self.greedy_split<(self.state.y+self.state.h):
				split_probs[self.greedy_split] = 1.-self.annealed_epsilon+self.annealed_epsilon/(self.state.h-1)

			split_probs/=split_probs.sum()

			self.state.boundaryscaled_split = npy.random.choice(range(self.data_loader.image_size-1),p=split_probs)
			self.state.split = float(self.state.boundaryscaled_split-self.state.y)/self.state.y

			# Transform to local co-ordinates. 
			self.state.boundaryscaled_split-=self.state.y
			state1 = parse_tree_node(label=0,x=self.state.x,y=self.state.y,w=self.state.w,h=self.state.boundaryscaled_split,backward_index=self.current_parsing_index)
			state2 = parse_tree_node(label=0,x=self.state.x,y=self.state.y+self.state.boundaryscaled_split,w=self.state.w,h=self.state.h-self.state.boundaryscaled_split,backward_index=self.current_parsing_index)			

		# This again is common to both states.
		state1.image_index = self.state.image_index		
		state2.image_index = self.state.image_index
		# Always inserting the lower indexed split first.
		self.insert_node(state1,self.current_parsing_index+1)
		self.insert_node(state2,self.current_parsing_index+2)

	def process_assignment(self):
		state1 = copy.deepcopy(self.parse_tree[self.current_parsing_index])
		state1.label = self.state.rule_applied
		state1.backward_index = self.current_parsing_index
		state1.image_index = self.state.image_index
		self.insert_node(state1,self.current_parsing_index+1)

	def parse_nonterminal_offpolicy(self):
		self.set_rule_mask()

		# Predict rule probabilities and select a rule from it IF epsilon.
		self.select_rule_behavioural_policy()
		
		if self.state.rule_applied==0 or self.state.rule_applied==1:
			# Function to process splits.	
			self.process_splits_behavioural_policy()

		elif self.state.rule_applied==2 or self.state.rule_applied==3:
			# Function to process assignments.
			self.process_assignment()	

	def parse_terminal(self):
		# Here, change value of predicted_labels.
		# Compute reward for state.
		# if self.state.label==1:
		# 	self.state.reward = self.data_loader.labels[self.state.image_index,self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h].sum()

		# elif self.state.label==2:
		# 	self.state.reward = -self.data_loader.labels[self.state.image_index,self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h].sum()	
		# embed()

		self.state.reward = ((-1)**(self.state.label-1))*(self.data_loader.labels[self.state.image_index,self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h].sum())
		self.predicted_labels[self.state.image_index,self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h] = self.state.label

	def construct_parse_tree(self, image_index):

		while ((self.predicted_labels[image_index]==0).any() or (self.current_parsing_index<=len(self.parse_tree)-1)):
			# embed()
			
			self.state = self.parse_tree[self.current_parsing_index]
			if self.state.label==0:
				self.parse_nonterminal_offpolicy()
			else:
				self.parse_terminal()

			self.current_parsing_index+=1

	def backward_tree_propagation(self):

		# Propagate rewards and likelihood ratios back up the tree.
		self.gamma = 1.

		for j in reversed(range(len(self.parse_tree))):	
			if (self.parse_tree[j].backward_index>=0):
				self.parse_tree[self.parse_tree[j].backward_index].reward += self.parse_tree[j].reward*self.gamma

		for j in range(len(self.parse_tree)):
			self.parse_tree[j].reward /= (self.parse_tree[j].w*self.parse_tree[j].h)
		
		if self.args.tanrewards:
			self.alpha = 1.0
			
			# Non-linearizing rewards.
			for j in range(len(self.parse_tree)):
				self.parse_tree[j].reward = npy.tan(self.alpha*self.parse_tree[j].reward)		

		# # Now propagating likelihood ratios.
		# for j in reversed(range(len(self.parse_tree))):
		# 	if (self.parse_tree[j].backward_index>=0):
		# 		self.parse_tree[self.parse_tree[j].backward_index].likelihood_ratio *= self.parse_tree[j].likelihood_ratio

	def backprop(self):
		self.batch_states = npy.zeros((self.batch_size,self.data_loader.image_size,self.data_loader.image_size,self.data_loader.num_channels))
		self.batch_target_rules = npy.zeros((self.batch_size,self.model.num_rules))
		self.batch_sampled_splits = npy.zeros((self.batch_size,1))
		self.batch_rule_masks = npy.zeros((self.batch_size,self.model.num_rules))
		self.batch_rule_weights = npy.zeros((self.batch_size,1))
		self.batch_split_weights = npy.zeros((self.batch_size,1))

		# Select indices of memory to put into batch.
		indices = self.memory.sample_batch()
		
		# Accumulate above variables into batches. 
		for k in range(len(indices)):
			state = copy.deepcopy(self.memory.memory[indices[k]])

			self.batch_states[k, state.x:state.x+state.w, state.y:state.y+state.h,0] = \
				self.data_loader.images[state.image_index, state.x:state.x+state.w, state.y:state.y+state.h]
			self.batch_rule_masks[k] = state.rule_mask
			if state.rule_applied==-1:
				self.batch_target_rules[k, state.rule_applied] = 0.
				self.batch_rule_weights[k] = 0.				
			else:
				self.batch_target_rules[k, state.rule_applied] = 1.
				# self.batch_rule_weights[k] = state.reward*state.likelihood_ratio
				self.batch_rule_weights[k] = state.reward
			if state.rule_applied==0:

				self.batch_sampled_splits[k] = state.split
				# self.batch_split_weights[k] = state.reward*state.likelihood_ratio
				self.batch_split_weights[k] = state.reward
		# embed()
		# Call sess train.
		self.sess.run(self.model.train, feed_dict={self.model.input: self.batch_states,
												   self.model.sampled_split: self.batch_sampled_splits,
												   self.model.split_return_weight: self.batch_split_weights,
												   self.model.target_rule: self.batch_target_rules,
												   self.model.rule_mask: self.batch_rule_masks,
												   self.model.rule_return_weight: self.batch_rule_weights})

	def meta_training(self,train=True):

		# Burn in memory. 
		self.predicted_labels = npy.zeros((self.data_loader.num_images,self.data_loader.image_size,self.data_loader.image_size))
		
		# embed()
		if self.args.train:
			self.burn_in()
			self.model.save_model(0)
		else:
			self.num_epochs=1	

		# For all epochs. 
		for e in range(self.num_epochs):
			self.average_episode_rewards = npy.zeros((self.data_loader.num_images))			
			self.predicted_labels = npy.zeros((self.data_loader.num_images,self.data_loader.image_size,self.data_loader.image_size,self.data_loader.num_channels))

			image_index_list = range(self.data_loader.num_images)
			npy.random.shuffle(image_index_list)

			# For all images in the dataset.
			for i in range(self.data_loader.num_images):
				
				# Initialize the tree for the current image.
				self.initialize_tree(image_index_list[i])

				# Set training parameters (Update epsilon).
				self.set_parameters(e)

				# Parse this image.
				self.construct_parse_tree(image_index_list[i])

				# Propagate rewards. 
				self.backward_tree_propagation()

				# Add to memory. 
				self.append_parse_tree()

				# Backprop --> over a batch sampled from memory. 
				self.backprop()
				print("Completed Epoch:",e,"Training Image:",i,"Total Reward:",self.parse_tree[0].reward)	

				self.average_episode_rewards[image_index_list[i]] = self.parse_tree[0].reward

			if self.args.train:
				# npy.save("predicted_labels_{0}.npy".format(e),self.predicted_labels)
				npy.save("rewards_{0}.npy".format(e),self.average_episode_rewards)
				if ((e%self.save_every)==0):
					self.model.save_model(e)				
			else: 
				npy.save("validation.npy",self.predicted_labels)
				npy.save("val_rewards.npy".format(e),self.average_episode_rewards)
			
			print("Cummulative Reward for Episode:",self.average_episode_rewards.mean())