#!/usr/bin/env python
from headers import *

# Define a class for the parse tree / rule / etc? 
class parse_tree_node():
	# def __init__(self, label=-1, x=-1, y=-1,w=-1,h=-1,backward_index=-1,rule_applied=-1, split=-1, start=npy.array([-1,-1]), goal=npy.array([-1,-1])):
	def __init__(self, label=-1, x=-1, y=-1,w=-1,h=-1,backward_index=-1,rule_applied=-1, split=0, start=npy.array([-1,-1]), goal=npy.array([-1,-1])):
		self.label = label
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.backward_index = backward_index
		self.rule_applied = rule_applied
		self.split = split
		self.reward = 0.
		self.start = start
		self.goal = goal

	def disp(self):
		print("Label:", self.label)
		print("X:",self.x,"Y:",self.y,"W:",self.w,"H:",self.h)
		print("Backward Index:",self.backward_index)
		print("Reward:",self.reward)
		print("Rule:",self.rule_applied,"Split:",self.split)
		print("____________________________________________")
