#!/usr/bin/env python
from headers import *
from state_class import parse_tree_node
import EntropySplits

class Parser():

	# In this class, we are going to learn assignments and splits.
	def __init__(self, model_instance=None, data_loader_instance=None, memory_instance=None, plot_manager=None, args=None, session=None): 

		self.ACModel = model_instance
		self.data_loader = data_loader_instance
		self.memory = memory_instance
		self.plot_manager = plot_manager
		self.args = args
		self.sess = session
		self.batch_size = 25
		self.num_epochs = 500
		self.save_every = 1
		self.max_parse_steps = 20
		self.minimum_width = 25
		self.max_depth = 7

		# Beta is probability of using expert.
		self.anneal_epochs = 100
		self.initial_beta = 1. 
		self.final_beta = 0.5
		self.beta_anneal_rate = (self.initial_beta-self.final_beta)/self.anneal_epochs

		self.initial_epsilon = 1e-3
		self.final_epsilon = 1e-3
		self.test_epsilon = 1e-4

		self.anneal_epsilon_rate = (self.initial_epsilon-self.final_epsilon)/self.anneal_epochs
		self.annealed_epsilon = copy.deepcopy(self.initial_epsilon)

	def initialize_tree(self,i):
		# Intialize the parse tree for this image.
		self.state = parse_tree_node(label=0,x=0,y=0,w=self.data_loader.image_size,h=self.data_loader.image_size)
		self.state.image_index = copy.deepcopy(i)
		self.current_parsing_index = 0
		self.parse_tree = [parse_tree_node()]
		self.parse_tree[self.current_parsing_index]=self.state

	def append_parse_tree(self):
		for k in range(len(self.parse_tree)):
			# Only adding non-terminal states to the memory. 
			# REMEMBER TO CHANGE THIS WHEN PAINTING.		

			# if self.parse_tree[k].depth==self.switch_depth-1:
			# 	if self.parse_tree[k].label==0:
			# 		self.memory.append_to_memory(self.parse_tree[k])

			if self.parse_tree[k].label==0:
				if self.parse_tree[k].depth>=self.switch_depth-1:
					self.memory.append_to_memory(self.parse_tree[k])

	def burn_in(self):
		
		# For one epoch, parse all images, store in memory.
		image_index_list = range(self.data_loader.num_images)
		npy.random.shuffle(image_index_list)

		self.burn_epochs = 1
		self.set_parameters(0)
	
		for e in range(self.burn_epochs):
			for i in range(self.data_loader.num_images):
				print("Epoch",e,"Burning in image:",i)
				# Initialize tree.
				self.initialize_tree(image_index_list[i])

				# Parse Image.
				self.construct_parse_tree(image_index_list[i])

				# Compute rewards.
				# self.compute_rewards()
				self.backward_tree_propagation()

				# For every state in the parse tree, push to memory.
				self.append_parse_tree()

	def set_parameters(self,e):

		# Setting parameters.
		if self.args.train:
			if e<self.anneal_epochs:
				self.annealed_epsilon = self.initial_epsilon-e*self.anneal_epsilon_rate
				self.annealed_beta = self.initial_beta-e*self.beta_anneal_rate			
			else:
				self.annealed_epsilon = self.final_epsilon
				self.annealed_beta = self.final_beta
				
		else:
			self.annealed_epsilon = self.test_epsilon	
			self.annealed_beta = self.final_beta

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

		self.state.rule_mask = npy.ones((4))

		if self.state.depth>=self.max_depth:
			# Allow only assignment.
			self.state.rule_mask[[0,1]] = 0.		

		if self.state.w<=self.minimum_width:
			self.state.rule_mask[0] = 0.

		if self.state.h<=self.minimum_width:		
			self.state.rule_mask[1] = 0.

	# If we have greedy rule and split selection for both the learner and the expert,
	# can derive an epsilon greedy version of either of these by acting randomly with probability epsilon. 
	# Only need to take care of valid random actions. 

	def select_rule_learner_greedy(self):
		# Remember, epsilon greedy is now outside of this. 
		# Constructing attended image.
		input_image = npy.zeros((1,self.data_loader.image_size,self.data_loader.image_size,self.data_loader.num_channels))

		# Change for 3 channel image.
		input_image[0,self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h,:] = \
			copy.deepcopy(self.data_loader.images[self.state.image_index,self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h])
		
		rule_probabilities = self.sess.run(self.ACModel.actor_network.rule_probabilities, feed_dict={self.ACModel.actor_network.input: input_image,
				self.ACModel.actor_network.rule_mask: self.state.rule_mask.reshape((1,self.ACModel.actor_network.num_rules))})

		# Don't need to change this, since rule mask is fed as input to the model. 
		self.state.rule_applied = npy.argmax(rule_probabilities)

	def select_rule_expert_greedy(self):
		
		# Greedily select a rule. 
		entropy_image_input = self.data_loader.labels[self.state.image_index,self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h]

		# Now returning both greedy axes and greedy splits. 
		self.greedy_axis,self.greedy_split = EntropySplits.best_valid_split(entropy_image_input, self.state.rule_mask)
		# EntropySplits takes care of the case where splits along one axis are invalid
		# but the other axis is valid .

		# If there was no information gain maximizing (or entropy reducing) split, 
		# or if all of the split rules were banned. 
		if (self.greedy_axis==-1) or (self.greedy_split==-1) or ((self.state.rule_mask[:2]==0).all()):
			# The sum value of the image is >0 for paint, <0 for non-paint.
			ip_img_sum = entropy_image_input.sum()
	
			if ip_img_sum>=0:
				# Now handling the =0 case as paint, so err to the side of painting. 
				# Good for imbalanced classes. 
				self.state.rule_applied = 2
			else:
				self.state.rule_applied = 3		

		else:
			# If there was a information gain maximizing split: 
			self.state.rule_applied = self.greedy_axis

	def select_rule_random(self):
		# For epsilon greedy policy derivation. 
		# Select a valid rule at random.
		self.state.rule_applied = npy.random.choice(npy.where(self.state.rule_mask)[0])

	def insert_node(self, state, index):
		# Put node into parse tree.
		self.parse_tree.insert(index,state)

	def select_split_expert_greedy(self):
		# Greedy split already in local patch coordinates. 
		self.state.boundaryscaled_split = copy.deepcopy(self.greedy_split)

		if self.state.rule_applied==0:
			# self.state.boundaryscaled_split = self.greedy_split + self.state.x
			# self.state.split = float(self.state.boundaryscaled_split-self.state.x)/self.state.w
			# self.state.boundaryscaled_split -= self.state.x
			self.state.split = float(self.state.boundaryscaled_split+self.state.x)
			# Must add resultant states to parse tree.
			state1 = parse_tree_node(label=0,x=self.state.x,y=self.state.y,w=self.state.boundaryscaled_split,h=self.state.h,backward_index=self.current_parsing_index)
			state2 = parse_tree_node(label=0,x=self.state.x+self.state.boundaryscaled_split,y=self.state.y,w=self.state.w-self.state.boundaryscaled_split,h=self.state.h,backward_index=self.current_parsing_index)

		if self.state.rule_applied==1:	
			# self.state.split = float(self.state.boundaryscaled_split)/self.state.h
			self.state.split = float(self.state.boundaryscaled_split+self.state.y)

			# Must add resultant states to parse tree.
			state1 = parse_tree_node(label=0,x=self.state.x,y=self.state.y,w=self.state.w,h=self.state.boundaryscaled_split,backward_index=self.current_parsing_index)
			state2 = parse_tree_node(label=0,x=self.state.x,y=self.state.y+self.state.boundaryscaled_split,w=self.state.w,h=self.state.h-self.state.boundaryscaled_split,backward_index=self.current_parsing_index)			

		# This again is common to both states.
		state1.image_index = self.state.image_index		
		state2.image_index = self.state.image_index
		state1.depth = self.state.depth+1
		state2.depth = self.state.depth+1
		# Always inserting the lower indexed split first.
		self.insert_node(state1,self.current_parsing_index+1)
		self.insert_node(state2,self.current_parsing_index+2)

	def select_split_learner_sample(self):
		# Constructing attended image.
		input_image = npy.zeros((1,self.data_loader.image_size,self.data_loader.image_size,self.data_loader.num_channels))
		input_image[0,self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h,:] = \
			copy.deepcopy(self.data_loader.images[self.state.image_index,self.state.x:self.state.x+self.state.w,self.state.y:self.state.y+self.state.h])

		if self.state.rule_applied==0:	
			self.state.split = self.sess.run(self.ACModel.actor_network.predicted_split,feed_dict={self.ACModel.actor_network.input: input_image,
				self.ACModel.actor_network.split_weight: npy.ones((1,1)),
				self.ACModel.actor_network.lower_lim: npy.reshape(self.state.x+1,(1,1)),
				self.ACModel.actor_network.upper_lim: npy.reshape(self.state.x+self.state.w-1,(1,1))})
			# embed()
			self.state.split = int(self.state.split[0])
			# embed()
			# print(float(self.state.split-self.state.x)/self.state.w)
			# Transform to local patch coordinates.
			self.state.boundaryscaled_split = self.state.split-self.state.x
			state1 = parse_tree_node(label=0,x=self.state.x,y=self.state.y,w=self.state.boundaryscaled_split,h=self.state.h,backward_index=self.current_parsing_index)
			state2 = parse_tree_node(label=0,x=self.state.x+self.state.boundaryscaled_split,y=self.state.y,w=self.state.w-self.state.boundaryscaled_split,h=self.state.h,backward_index=self.current_parsing_index)

		if self.state.rule_applied==1:
			self.state.split = self.sess.run(self.ACModel.actor_network.predicted_split,feed_dict={self.ACModel.actor_network.input: input_image,
				self.ACModel.actor_network.split_weight: npy.ones((1,1)),
				self.ACModel.actor_network.lower_lim: npy.reshape(self.state.y+1,(1,1)),
				self.ACModel.actor_network.upper_lim: npy.reshape(self.state.y+self.state.h-1,(1,1))})
			# embed()
			self.state.split = int(self.state.split[0])
			# print(float(self.state.split-self.state.y)/self.state.h)
			# Transform to local patch coordinates.
			self.state.boundaryscaled_split = self.state.split- self.state.y
			state1 = parse_tree_node(label=0,x=self.state.x,y=self.state.y,w=self.state.w,h=self.state.boundaryscaled_split,backward_index=self.current_parsing_index)
			state2 = parse_tree_node(label=0,x=self.state.x,y=self.state.y+self.state.boundaryscaled_split,w=self.state.w,h=self.state.h-self.state.boundaryscaled_split,backward_index=self.current_parsing_index)			

		# Must add resultant states to parse tree.
		state1.image_index = self.state.image_index		
		state2.image_index = self.state.image_index
		state1.depth = self.state.depth+1
		state2.depth = self.state.depth+1
		# Always inserting the lower indexed split first.
		self.insert_node(state1,self.current_parsing_index+1)
		self.insert_node(state2,self.current_parsing_index+2)

	def process_assignment(self):
		state1 = copy.deepcopy(self.parse_tree[self.current_parsing_index])
		state1.depth = self.state.depth+1
		state1.label = self.state.rule_applied-1
		state1.backward_index = self.current_parsing_index
		state1.image_index = self.state.image_index
		self.insert_node(state1,self.current_parsing_index+1)

	def parse_nonterminal_expert(self):
		self.set_rule_mask()

		# Predict rule probabilities and select a rule from it IF epsilon.
		# self.select_rule_behavioural_policy()
		self.select_rule_expert_greedy()
		
		if self.state.rule_applied==0 or self.state.rule_applied==1:
			# Function to process splits.	
			# self.process_splits_behavioural_policy()
			self.select_split_expert_greedy()

		elif self.state.rule_applied==2 or self.state.rule_applied==3:
			# Function to process assignments.
			self.process_assignment()	

	def parse_nonterminal_learner(self):
		self.set_rule_mask()

		# Predict rule probabilities and select a rule from it IF epsilon.
		if npy.random.random()<self.annealed_epsilon:
			self.select_rule_random()
		else:
			self.select_rule_learner_greedy()

		if self.state.rule_applied==0 or self.state.rule_applied==1: 
			# Function to process splits.	
			self.select_split_learner_sample()

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
		
		# Sample a depth beyond which we follow the expert policy deterministically. 
		# This switch depth is true for the entire parse - i.e. all branches of the parse tree. 
		self.switch_depth = npy.random.randint(1,high=self.max_depth+1)

		while ((self.predicted_labels[image_index]==0).any() or (self.current_parsing_index<=len(self.parse_tree)-1)):					

			self.state = self.parse_tree[self.current_parsing_index]
			if self.state.label==0:

				if self.args.train:
					if self.state.depth<self.switch_depth:

						random_probability = npy.random.random()

						if random_probability<self.annealed_beta:
							self.parse_nonterminal_expert()
						else:
							self.parse_nonterminal_learner()
					else:
						self.parse_nonterminal_expert()				
				else:
					# Using Learnt Policy
					# self.parse_nonterminal_expert()
					self.parse_nonterminal_learner()	

			else:
				self.parse_terminal()
			
			if self.args.plot:
				self.plot_manager.update_plot_data(image_index, self.predicted_labels[image_index], self.parse_tree, self.current_parsing_index)

			self.current_parsing_index+=1

	def backward_tree_propagation(self):

		# Propagate rewards and likelihood ratios back up the tree.
		self.gamma = 1.

		for j in reversed(range(len(self.parse_tree))):	
			if (self.parse_tree[j].backward_index>=0):
				self.parse_tree[self.parse_tree[j].backward_index].reward += self.parse_tree[j].reward*self.gamma

		# if self.args.train:
		# 	for j in range(len(self.parse_tree)):
		# 		self.parse_tree[j].reward *= 10./(self.parse_tree[j].w*self.parse_tree[j].h)
		# else:
		for j in range(len(self.parse_tree)):
			self.parse_tree[j].reward /= (self.parse_tree[j].w*self.parse_tree[j].h)

		# # Scaling rewards by constant value - image_size **2 .(so it's at maximum 1).
		# for j in range(len(self.parse_tree)):
		#	self.parse_tree[j].reward *= 10./(self.data_loader.image_size**2)

		# for j in range(len(self.parse_tree)):
		# 	self.parse_tree[j].reward /= (self.parse_tree[j].w*self.parse_tree[j].h)

		# # # Scaling rewards by constant value - image_size **2 .(so it's at maximum 1).
		# for j in range(len(self.parse_tree)):
		# 	self.parse_tree[j].reward *= 10./(self.data_loader.image_size**2)


	def backprop(self, iter_num):
		self.batch_states = npy.zeros((self.batch_size,self.data_loader.image_size,self.data_loader.image_size,self.data_loader.num_channels))
		self.batch_rule_masks = npy.zeros((self.batch_size,self.ACModel.actor_network.num_rules))
		self.batch_split_weights = npy.zeros((self.batch_size,1))
		self.batch_lower_lims = npy.zeros((self.batch_size,1))
		self.batch_upper_lims = npy.ones((self.batch_size,1))
		self.batch_target_Qvalues = npy.zeros((self.batch_size,1))

		self.batch_target_rules = npy.zeros((self.batch_size,self.ACModel.actor_network.num_rules))
		self.batch_rule_weights = npy.zeros((self.batch_size,1))

		# Select indices of memory to put into batch.
		indices = self.memory.sample_batch()
		
		# Accumulate above variables into batches. 
		for k in range(len(indices)):

			state = copy.deepcopy(self.memory.memory[indices[k]])

			self.batch_states[k, state.x:state.x+state.w, state.y:state.y+state.h,:] = \
				self.data_loader.images[state.image_index, state.x:state.x+state.w, state.y:state.y+state.h]
			# self.batch_rule_masks[k] = state.rule_mask

			if state.rule_applied==-1:
				self.batch_target_rules[k, state.rule_applied] = 0.
				self.batch_rule_weights[k] = 0.				
			else:
				self.batch_target_rules[k, state.rule_applied] = 1.
				self.batch_rule_weights[k] = state.reward
				self.batch_target_Qvalues[k] = state.reward
				self.batch_rule_masks[k] = state.rule_mask

			if state.rule_applied==0 or state.rule_applied==1:						
				self.batch_split_weights[k] = 1.

				if state.rule_applied==0:
					self.batch_lower_lims[k] = float(state.x+1)
					self.batch_upper_lims[k] = float(state.x+state.w-1)
				if state.rule_applied==1:
					self.batch_lower_lims[k] = float(state.y+1)
					self.batch_upper_lims[k] = float(state.y+state.h-1)
		
		# if iter_num%20==0:
		# embed()
		# MAYBE UPDATE CRITIC FIRST, BECAUSE OTHERWISE THE ACTION "TAKEN" Changes? 
		merged, _, _ = self.sess.run([self.ACModel.merged_summaries, self.ACModel.train_critic, self.ACModel.train_actor], feed_dict={self.ACModel.actor_network.input: self.batch_states,
			self.ACModel.actor_network.split_weight: self.batch_split_weights,
			self.ACModel.actor_network.lower_lim: self.batch_lower_lims,
			self.ACModel.actor_network.target_rule: self.batch_target_rules,
			self.ACModel.actor_network.rule_return_weight: self.batch_rule_weights,
			self.ACModel.actor_network.upper_lim: self.batch_upper_lims,
			self.ACModel.actor_network.rule_mask: self.batch_rule_masks,
			self.ACModel.critic_network.input: self.batch_states,
			self.ACModel.target_Qvalue: self.batch_target_Qvalues})

		self.ACModel.tf_writer.add_summary(merged, iter_num)		

	def epoch_setup(self,e):
		self.average_episode_rewards = npy.zeros((self.data_loader.num_images))			
		self.predicted_labels = npy.zeros((self.data_loader.num_images,self.data_loader.image_size,self.data_loader.image_size))

		self.image_index_list = range(self.data_loader.num_images)
		npy.random.shuffle(self.image_index_list)

		# Set training parameters (Update epsilon).
		self.set_parameters(e)

	def save_setup(self,e):
		if self.args.train:
			npy.save("rewards_{0}.npy".format(e),self.average_episode_rewards)
			if ((e%self.save_every)==0):				
				self.ACModel.save_model(e)
		else: 
			npy.save("val_rewards_{0}.npy".format(self.args.suffix),self.average_episode_rewards)

	def meta_training(self,train=True):

		# Burn in memory. 
		self.predicted_labels = npy.zeros((self.data_loader.num_images,self.data_loader.image_size,self.data_loader.image_size))
		
		# embed()
		if self.args.train:
			self.burn_in()
			self.ACModel.save_model(0)
		else:
			self.num_epochs=1	

		# For all epochs. 
		for e in range(self.num_epochs):

			self.epoch_setup(e)

			# For all images in the dataset.
			for i in range(self.data_loader.num_images):
				
				# Initialize the tree for the current image.
				self.initialize_tree(self.image_index_list[i])

				# Parse this image.
				self.construct_parse_tree(self.image_index_list[i])

				# Propagate rewards. 
				self.backward_tree_propagation()

				# Add to memory. 
				self.append_parse_tree()
				
				# Backprop --> over a batch sampled from memory. 
				if self.args.train:
					# if e==0 and i>50:
					# 	self.backprop(self.data_loader.num_images*e+i)
					# if e>0:
					# 	self.backprop(self.data_loader.num_images*e+i)

					self.backprop(self.data_loader.num_images*e+i)					

				print("Completed Epoch:",e,"Training Image:",i,"Total Reward:",self.parse_tree[0].reward)	

				self.average_episode_rewards[self.image_index_list[i]] = self.parse_tree[0].reward

			self.save_setup(e)
			print("Cummulative Reward for Episode:",self.average_episode_rewards.mean())
