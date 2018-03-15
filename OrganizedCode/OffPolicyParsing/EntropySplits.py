#!/usr/bin/env python
from headers import *

# def infogain(image_segment,location,axis=0):
def infogain(image_segment,location):
	# Defining a function to calculate the information gain by splitting
	# at a particular location along a particular axis. 

	# # In general, must check the axes.
	# if axis==0:
	# 	s1 = image_segment[:location,:]
	# 	s2 = image_segment[location:,:]
	# if axis==1:
	# 	s1 = image_segment[:,:location]
	# 	s2 = image_segment[:,location:]

	# Now we can just assume that we're being provided a horizontal split location.
	segment1 = image_segment[:location,:]
	segment2 = image_segment[location:,:]

	segment_size = image_segment.shape[0]*image_segment.shape[1]
	probability_segment1 = float(segment1.shape[0]*segment1.shape[1])/segment_size
	probability_segment2 = float(segment2.shape[0]*segment2.shape[1])/segment_size

	return entropy(y)-(probability_segment1*entropy(segment1)+probability_segment2*entropy(segment2))

def entropy(image_segment):
	# Calculate the emperical entropy for an image segment. 
	nones = npy.count_nonzero(image_segment)

	segment_size = image_segment.shape[0]*image_segment.shape[1]

	nzeros = segment_size-nones

	p_zeros = float(nzeros)/segment_size
	p_ones = float(nones)/segment_size

	if p_zeros==0 or p_ones==0: 
		return 0.                

	return -(p_zeros*npy.log2(p_zeros)+p_ones*npy.log2(p_ones))  

def bestsplit(image_segment):

	# maxval = -1
	# chosen_a = -1
	# chosen_l = -1

	# # In general, iterate over both axes.
	# for a_val in range(2):
	# 	if a_val==0:
	# 		limval = image_segment.shape[0]-1
	# 	if a_val==1:
	# 		limval = image_segment.shape[1]-1
	# 	for l_val in range(1,limval):
			
	# 		ig = infogain(image_segment,a_val,l_val)
	# 		if ig>maxval:
	# 			maxval=ig
	# 			chosen_a = a_val
	# 			chosen_l = l_val
	# if maxval==0:
	# 	print("No entropy reducing splits.")
	# 	return 1
	# return chosen_a,chosen_l

	# For now, we can assume horizontal axis. 
	maxval = -1
	chosen_l = -1

	limval = image_segment.shape[0]-1
	# Iterate over all the locations.
	for l_val in range(1,limval):
		ig = infogain(image_segment,l_val)
		if ig>maxval:
			maxval=ig
			chosen_l=l_val
	if maxval==0:
		print("No entropy reducing splits.")
		return -1
	return chosen_l

# No need of this now.
# def return_splits(x,a,l):
# 	if a==0:
# 		s1=x[:l,:]
# 		s2=x[l:,:]
# 	if a==1:
# 		s1=x[:,:l]
# 		s2=x[:,l:]
# 	return s1,s2

