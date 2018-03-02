#!/usr/bin/env python
from headers import *

class Replay_Memory():

	def __init__(self, memory_size=10000):

		self.memory = []
		
		self.memory_size = memory_size

		# Essentially stores everything we need for backprop.
		# Inputs --> images, for which we need state and boundaries.
		# Sampled split. 
		# Rule applied (target rule).
		# Split weight (return).
		# Rule weight.

	def append_to_memory(self, transition):

		memory_len = len(self.memory)

		if memory_len<self.memory_size:
			self.memory.append(transition)
		else:
			self.memory.pop(0)
			self.memory.append(transition)

	def sample_batch(self, batch_size=32):
		
		memory_len = len(self.memory)	
		indices = npy.random.randint(0,high=memory_len,size=(batch_size))

		# The data loader is really what is storing the images. 
		# Instead of creating a batch of data (such as image inputs etc.) here,
		# pass indices back to parsing code. 
		# Do the actual batching there, so that we don't have to pass data_loader here. 
		return indices