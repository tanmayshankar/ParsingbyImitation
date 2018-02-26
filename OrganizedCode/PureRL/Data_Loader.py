#!/usr/bin/env python
from headers import *

class DataLoader():

	def __init__(self, image_path, label_path, indices_path=None, rewards_path=None):

		self.images = npy.load(image_path)
		self.labels = npy.load(label_path)

		if rewards_path:
			self.horizontal_rewards = npy.load(rewards_path)
			
		if indices_path:
			self.selected_indices = npy.load(indices_path)			
			self.images = self.images[self.selected_indices]
			self.labels = self.labels[self.selected_indices]
			self.horizontal_rewards = self.horizontal_rewards[self.selected_indices]
		
		if rewards_path:
			self.horizontal_rewards = self.horizontal_rewards.max(axis=(1,2))
		
		self.num_images = self.images.shape[0]
		self.image_size = self.images.shape[1]

	def preprocess(self):

		# For RGB Images:
		if len(self.images.shape)==4:

			for i in range(self.num_images):
				self.images[i] = cv2.cvtColor(self.images[i],cv2.COLOR_RGB2BGR)

			self.images = self.images.astype(float)
			self.images -= self.images.mean(axis=(0,1,2))

		# elif len(self.images.shape)==3:
			# self.images_3channel = npy.zeros((self.num_images,self.image_size,self.image_size,3))

			# for i in range(self.num_images):
			# 	for k in range(3):
			# 		self.images_3channel[i,:,:,k] = self.images[i]
		
			# self.images_3channel = self.images_3channel*127
			# self.images_3channel += 127
			# self.images_3channel -= self.images_3channel.mean(axis=(0,1,2))

			# Since we are not using Keras for this, Don't mean normalize or convert to 3 channel.
