#!/usr/bin/env python
from headers import *


# FOR ALL EPOCHS.
# FOR ALL IMAGES.
# Initialize Parse Tree and Image. 
# While the image still has non-terminal labelled pixels.
	# Select the current non-terminal from the parse tree.
	# (Select the next element in the parse tree).
	
	# If it's a non-terminal:       
		# Forward pass rule-policy network: obtain softmax over rules.
		# Sample this softmax probability distribution (Categorical distribution)
		# If the rule is a split rule:
		
			# Forward pass split-location network: sample split location from Gaussian with predicted mean/cov. 
			# Apply sampled rule with split location. 
			# Add splits to parse tree. 
	
	# If it's a terminal: 
		# Choose a goal location: sample goal from 2D Gaussian with predicted mean/cov.
	
# Remember, we are going to use the following grammar:
	############################
	# Rule numbers:
	# 0 (Shape) -> (Shape)(Shape) 								(Vertical split)
	# 1 (Shape) -> (Shape)(Shape) 								(Horizontal split)
	# 2 (Shape) -> (Shape)(Shape) 								(Vertical split with opposite order: top-bottom expansion)
	# 3 (Shape) -> (Shape)(Shape) 								(Horizontal split with opposite order: right-left expansion)
	# 4 (Shape) -> (Region with primitive #) 
	# 5 (Shape) -> (Region not to be painted)
	############################

# With split locations always at half positions, and with a minimum split width of 4

class parse_tree_node():
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

	def disp(self):
		print("Label:", self.label)
		print("X:",self.x,"Y:",self.y,"W:",self.w,"H:",self.h)

		print("Backward Index:",self.backward_index)
		print("Reward:",self.reward)
		print("Rule:",self.rule_applied,"Split:",self.split)
		print("____________________________________________")

num_images = 20000

# image_size = 50
# minimum_split_width = 5
# max_width = 30

image_size = 20
minimum_split_width = 3
max_width = 18

images = npy.zeros((num_images,image_size,image_size))

# For all images:
for i in range(num_images):
	
	# Initialize for this image.
	current_parsing_index = 0
	parse_tree = [parse_tree_node()]
	state = parse_tree_node(label=0,x=0,y=0,w=image_size,h=image_size)
	parse_tree[current_parsing_index]=state
	
	print("Processing image",i)
	while (images[i]==0).any():
		state = parse_tree[current_parsing_index]
		if state.label==0:
			rule_probs = npy.ones((6))
			# state.disp()
			if (state.h<=minimum_split_width):
				rule_probs[[0,2]]=0.
			if (state.w<=minimum_split_width):
				rule_probs[[1,3]]=0.
			if (state.w>=max_width) or (state.h>=max_width):
				rule_probs[[4,5]]=0.
			
			rule_probs/=rule_probs.sum()
			rule = npy.random.choice(range(6),p=rule_probs)
			
			if rule<=3:
				# Now must expand.
				# If vertical split rule.
				if rule==0 or rule==2:
					# split = int(float(state.h)/2)
					# split = npy.random.choice(range(image_size))

					split = npy.random.choice(range(minimum_split_width,image_size - minimum_split_width))

					# print("PRE:",split)
					if split>=image_size/2:
						split = int(npy.floor(float(state.h*split)/image_size))						
					else:
						split = int(npy.ceil(float(state.h*split)/image_size))		
					# print("POST:",split)
					s1 = parse_tree_node(label=0,x=state.x,y=state.y,w=state.w,h=split,backward_index=current_parsing_index)
					s2 = parse_tree_node(label=0,x=state.x,y=state.y+split,w=state.w,h=state.h-split,backward_index=current_parsing_index)

				if rule==1 or rule==3:
					
					# print("PRE:",split)
					# split = npy.random.choice(range(image_size))
					
					split = npy.random.choice(range(minimum_split_width,image_size - minimum_split_width))
					if split>=image_size/2:
						split = int(npy.floor(float(state.w*split)/image_size))
					else:
						split = int(npy.ceil(float(state.w*split)/image_size))		
					# print("POST:",split)
					s1 = parse_tree_node(label=0,x=state.x,y=state.y,w=split,h=state.h,backward_index=current_parsing_index)
					s2 = parse_tree_node(label=0,x=state.x+split,y=state.y,w=state.w-split,h=state.h,backward_index=current_parsing_index)

				if rule<=1:
					parse_tree.insert(current_parsing_index+1,s1)
					parse_tree.insert(current_parsing_index+2,s2)
				if rule>=2:
					parse_tree.insert(current_parsing_index+1,s2)
					parse_tree.insert(current_parsing_index+2,s1)

			if rule>=4:
				# Don't actually need to add terminals to parse tree.
				s1 = copy.deepcopy(parse_tree[current_parsing_index])
				s1.label = rule-3
				images[i,s1.x:s1.x+s1.w,s1.y:s1.y+s1.h] = s1.label
				parse_tree.insert(current_parsing_index+1,s1)

		current_parsing_index+=1

npy.save("Images_20_general.npy",images)