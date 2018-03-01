#!/usr/bin/env python
from headers import *

class Parser():

	# In this class, we are going to learn assignments and 
	def __init__(self, model_instance=None, data_loader_instance=None, args=None, session=None):

		self.model = model_instance
		self.data_loader = data_loader_instance
		self.args = args
		self.sess = session
		self.batch_size = 25
		self.num_epochs = 250
		self.save_every = 1


		# Parameters for annealing covariance. 
		self.initial_cov = 0.1
		self.final_cov = 0.01
		self.anneal_epochs = 40
		self.anneal_rate = (self.initial_cov-self.final_cov)/self.anneal_epochs

	def forward(self, indices):
		# Forward pass of the network.      
		# Now we are going to force a split as the first action. 
		# Then we select a split location from the network. 
		# Create two splits. 
		# For each split, pick an assignment. 
		# Evaluate the reward.
		# Backprop this. 

		# Creating an array of indices.
		num_images_batch = len(indices)
		indices = npy.array(indices)

		# For ensuring splits are inside the current segment.
		redo_indices = npy.ones((num_images_batch)).astype(bool)
		self.batch_sampled_splits = npy.zeros((num_images_batch))

		while redo_indices.any():

			if self.args.anneal_cov:
				self.batch_sampled_splits[redo_indices] = self.sess.run(self.model.sample_split,
					feed_dict={self.model.input: self.data_loader.images[indices[redo_indices]].reshape((npy.count_nonzero(redo_indices), self.model.image_size, self.model.image_size,self.model.num_channels)),
							   self.model.split_cov: self.covariance_value })[:,0]          
			else:
				self.batch_sampled_splits[redo_indices] = self.sess.run(self.model.sample_split,
					feed_dict={self.model.input: self.data_loader.images[indices[redo_indices]].reshape((npy.count_nonzero(redo_indices), self.model.image_size, self.model.image_size,self.model.num_channels)) })[:,0]

			redo_indices = (self.batch_sampled_splits<0.)+(self.batch_sampled_splits>1.)

		# We are rescaling the image to 1 to 255.
		# Putting this in a different vector because we need to backprop with the original.
		self.rescaled_batch_sampled_splits = (self.batch_sampled_splits*(self.model.image_size-1)+1.).astype(int)

		# Now that we've sampled splits, we must select assignments for\ both sets of segments
		# For each of the images.

		self.batch_sampled_rules = npy.zeros((num_images_batch, 2))

		# Create attended images.
		self.attended_images_first = npy.zeros((num_images_batch, self.model.image_size, self.model.image_size, self.model.num_channels))
		self.attended_images_second = npy.zeros((num_images_batch, self.model.image_size, self.model.image_size, self.model.num_channels))

		# Copying over original image content into the attended images.
		for j in range(num_images_batch):
			self.attended_images_first[j,:self.rescaled_batch_sampled_splits[j],:,:] = self.data_loader.images[indices[j],:self.rescaled_batch_sampled_splits[j],:].reshape((self.rescaled_batch_sampled_splits[j],self.model.image_size,1))
			self.attended_images_second[j,self.rescaled_batch_sampled_splits[j]:,:,:] = self.data_loader.images[indices[j],self.rescaled_batch_sampled_splits[j]:,:].reshape((self.model.image_size-self.rescaled_batch_sampled_splits[j],self.model.image_size,1))

		# SAY RULE 0 --> PAINT
		# RULE 1 --> NOT PAINT
		self.batch_sampled_rules[:,0] = self.sess.run(self.model.sampled_rule,
			feed_dict={self.model.input: self.attended_images_first})
		self.batch_sampled_rules[:,1] = self.sess.run(self.model.sampled_rule,
			feed_dict={self.model.input: self.attended_images_second})	

		self.seg0_reward_values = npy.zeros((num_images_batch))
		self.seg1_reward_values = npy.zeros((num_images_batch))
		self.split_reward_values = npy.zeros((num_images_batch))

		for j in range(num_images_batch):
			self.painted_images[indices[j],:self.rescaled_batch_sampled_splits[j],:] = 1.-2*self.batch_sampled_rules[j,0]

			# a = self.painted_images[indices[j],:self.rescaled_batch_sampled_splits[j],:]
			# b = self.data_loader.labels[indices[j],:self.rescaled_batch_sampled_splits[j],:]
			# embed()
			self.seg0_reward_values[j] = (self.painted_images[indices[j],:self.rescaled_batch_sampled_splits[j],:]*self.data_loader.labels[indices[j],:self.rescaled_batch_sampled_splits[j],:]).mean()

			self.painted_images[indices[j],self.rescaled_batch_sampled_splits[j]:,:] = 1.-2*self.batch_sampled_rules[j,1]
			self.seg1_reward_values[j] = (self.painted_images[indices[j],self.rescaled_batch_sampled_splits[j]:,:]*self.data_loader.labels[indices[j],self.rescaled_batch_sampled_splits[j]:,:]).mean()

	def compute_reward(self, indices):
		# reward_values = (self.painted_images[indices]*self.data_loader.labels[indices]).mean(axis=(1,2))

		# Compute rewards for all three things.
		self.split_reward_values = (self.painted_images[indices]*self.data_loader.labels[indices]).mean(axis=(1,2))

		self.rewards[indices,0] = self.split_reward_values
		self.rewards[indices,1] = self.seg0_reward_values
		self.rewards[indices,2] = self.seg1_reward_values
	
	def backprop(self, indices):        
		# Updating the network.

		num_images_batch = len(indices)

		# Return weights start to matter here. 		
		# First for splits with entire images.
		if self.args.anneal_cov:
			self.sess.run(self.model.train, feed_dict={self.model.input: self.data_loader.images[indices].reshape((num_images_batch, self.model.image_size, self.model.image_size,self.model.num_channels)) ,
													   self.model.sampled_split: self.batch_sampled_splits.reshape((num_images_batch,1)),
													   self.model.split_return_weight: self.split_reward_values.reshape((num_images_batch,1)),
													   self.model.split_cov: self.covariance_value, 
													   self.model.target_rule: npy.zeros((num_images_batch)),
													   self.model.rule_return_weight: npy.zeros((num_images_batch,1))})        

			# Next for segment zeros. 
			self.sess.run(self.model.train, feed_dict={self.model.input: self.attended_images_first,
													   self.model.sampled_split: npy.zeros((num_images_batch,1)),
													   self.model.split_return_weight: npy.zeros((num_images_batch,1)),
													   self.model.split_cov: self.covariance_value, 
													   self.model.target_rule: self.batch_sampled_rules[:,0],
													   self.model.rule_return_weight: self.seg0_reward_values.reshape((num_images_batch,1))})        

			# For segment ones.
			self.sess.run(self.model.train, feed_dict={self.model.input: self.attended_images_second,
													   self.model.sampled_split: npy.zeros((num_images_batch,1)),
													   self.model.split_return_weight: npy.zeros((num_images_batch,1)),
													   self.model.split_cov: self.covariance_value, 
													   self.model.target_rule: self.batch_sampled_rules[:,1],
													   self.model.rule_return_weight: self.seg1_reward_values.reshape((num_images_batch,1))})        
		else:
			self.sess.run(self.model.train, feed_dict={self.model.input: self.data_loader.images[indices].reshape((num_images_batch, self.model.image_size, self.model.image_size,self.model.num_channels)) ,
													   self.model.sampled_split: self.batch_sampled_splits.reshape((num_images_batch,1)),
													   self.model.split_return_weight: self.split_reward_values.reshape((num_images_batch,1)),
													   self.model.target_rule: npy.zeros((num_images_batch)),
													   self.model.rule_return_weight: npy.zeros((num_images_batch,1))})        

			# Next for segment zeros. 
			self.sess.run(self.model.train, feed_dict={self.model.input: self.attended_images_first,
													   self.model.sampled_split: npy.zeros((num_images_batch,1)),
													   self.model.split_return_weight: npy.zeros((num_images_batch,1)),
													   self.model.target_rule: self.batch_sampled_rules[:,0],
													   self.model.rule_return_weight: self.seg0_reward_values.reshape((num_images_batch,1))})        

			# For segment ones.
			self.sess.run(self.model.train, feed_dict={self.model.input: self.attended_images_second,
													   self.model.sampled_split: npy.zeros((num_images_batch,1)),
													   self.model.split_return_weight: npy.zeros((num_images_batch,1)),
													   self.model.target_rule: self.batch_sampled_rules[:,1],
													   self.model.rule_return_weight: self.seg1_reward_values.reshape((num_images_batch,1))})        

	def set_covariance_value(self,e):
		if e<self.anneal_epochs:
			self.covariance_value = self.initial_cov - self.anneal_rate*e
		else:
			self.covariance_value = self.final_cov
		print("Setting covariance as:",self.covariance_value)

	def meta_training(self,train=True):
		
		image_index = 0
		self.painted_images = -npy.ones((self.data_loader.num_images, self.model.image_size,self.model.image_size))
		self.rewards = npy.zeros((self.data_loader.num_images,3))

		if self.args.plot:		
			self.define_plots()

		if not(self.args.train):
			self.num_epochs=1

		print("Entering Training Loops.")
		for e in range(self.num_epochs):
	
			index_list = range(self.data_loader.num_images)
			npy.random.shuffle(index_list)

			if self.args.train:
				if self.args.anneal_cov:
					self.set_covariance_value(e)
				else:
					self.covariance_value = 0.05
			else:
				self.covariance_value = 0.001

			for i in range(self.data_loader.num_images/self.batch_size):		
				indices = index_list[i*self.batch_size:min(self.data_loader.num_images,(i+1)*self.batch_size)]				
				# Forward pass over these images.
				self.forward(indices)
				self.compute_reward(indices)


				print("#__________________________________________")
				print("Epoch:",e,"Training Batch:",i)				

				# Backward pass if we are training.				
				if self.args.train:
					self.backprop(indices)
			
			print("AFTER EPOCH:",e,"AVERAGE REWARD:",self.rewards[:,0].mean())		
			if self.args.train:
				npy.save("painted_images_{0}.npy".format(e),self.painted_images)
				npy.save("rewards_{0}.npy".format(e),self.rewards)
				if ((e%self.save_every)==0):
					self.model.save_model(e)				
			else: 
				npy.save("validation.npy",self.painted_images)
				npy.save("val_rewards.npy",self.rewards)

			self.painted_images = -npy.ones((self.data_loader.num_images, self.model.image_size,self.model.image_size))
			self.rewards = npy.zeros((self.data_loader.num_images,3))	
