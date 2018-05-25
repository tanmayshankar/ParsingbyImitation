#!/usr/bin/env python
from headers import *

class Replay_Memory():

	def __init__(self, memory_size=2000):

		self.memory = []
		
		self.memory_size = memory_size

		# Essentially stores everything we need for backprop.
		# Inputs --> images, for which we need state and boundaries.
		# Sampled split. 
		# Rule applied (target rule).
		# Split weight (return).
		# Rule weight.

	def append_to_memory(self, state, k, parse_tree):
		# The neat thing is now the "State" class is sufficient - no need to explicitly provide
		# transitions as is typical of a memory. State class members have rule / split / masks. 

		memory_len = len(self.memory)

		if memory_len<self.memory_size:
			self.memory.append([state,k,parse_tree])
		else:
			self.memory.pop(0)
			self.memory.append([state,k,parse_tree])

	def sample_batch(self, batch_size=5):
		
		memory_len = len(self.memory)	
		indices = npy.random.randint(0,high=memory_len,size=(batch_size))
		# indices = range(0,batch_size)

		# for k in range(batch_size):
		# 	self.memory.pop(k)

		# The data loader is really what is storing the images. 
		# Instead of creating a batch of data (such as image inputs etc.) here,
		# pass indices back to parsing code. 
		# Do the actual batching there, so that we don't have to pass data_loader here. 
		# (Because transitions, i.e. state etc. only have indices / coordinates, not the actual image values.)
		return indices