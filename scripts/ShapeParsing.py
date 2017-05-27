
# coding: utf-8

# In[1]:

import numpy as npy
import matplotlib.pyplot as plt
import copy


# In[22]:

num_images = 200
images = 3*npy.ones((num_images,10,10))
labels = npy.zeros((num_images,10,10))


# In[23]:

# Initializing the rules of the grammar. 
# Description: 

# LABELS: 
# Facade - 
3n# Floor - 2
# Wall - 1
# Window - 0

# (0) Facade -> Floor Facade (horizontal) : #3 -> #2 #3
# (1) Floor -> Wall (no split)            : #2 -> #1
# (2) Floor -> Wall Floor (vertical)      : #2 -> #1 #2
# (3) Floor -> Window Floor (vertical)    : #2 -> #0 #2
# (4) Floor -> Window (no split)          : #2 -> #0

# Thus the current nonterminal may be labelled as: Facade or a Floor 
# Output labels must be Wall or Window (0 and 1)


# In[24]:

# Here's how we will implement the tree:
# LIST: [ [State, indexes it leads to], ...  ]

parse_tree = [[] for i in range(num_images)]
rules_applied = [[] for i in range(num_images)]

rewards = [[] for i in range(num_images)]
backward_indices =[[] for i in range(num_images)]


# In[25]:

state = [[3,0,0,10,10],[0,0]]
state[0]


# In[26]:

# Should initialize images created from these grammars. 
for i in range(1):
	
	# Set the state as the entire image: First 5 are state, next 2 are indices
	# State: [c,x,y,W,H]
	state = [3,0,0,10,10]
		
	parse_tree[i].append(state) 
	rewards[i].append(0)
	backward_indices[i].append([-1])
	
	current_parsing_index = 0
	rule_index = 0
	# While there are 2's and 3's in the images. 
	while ((images[i]==2).any() or (images[i]==3).any()):
			   
		state = parse_tree[i][current_parsing_index]
		print("________________________________")               
		print("State:",state)
		
		# If it's a facade, split into Floor and Facade from y to y+H             
		if state[0]==3:
			
			if state[4]==1:
				s1 = copy.deepcopy(state)
				s1[0]=2
				parse_tree[i].insert(current_parsing_index+1,s1)
				
			else: 
#                 ysplit = npy.random.randint(1,high=state[4]) #Add this to the state index 
				ysplit = npy.random.random_integers(1,high=state[4]) #Add this to the state index 
			# First a floor
			s1 = [2,state[1],state[2],state[3],ysplit]
			# The remainder is a facade, don't change label
			s2 = [3,state[1],state[2]+ysplit,state[3],state[4]-ysplit]
						
#             parse_tree[i][current_parsing_index][1][0] = index_to_insert
#             parse_tree[i][current_parsing_index][1][1] = index_to_insert+1
#             index_to_insert+=2

			# MUST NOW INSERT s1 and s2 INTO THE TREE JUST AFTER CURRENT PARSING INDEX
			parse_tree[i].insert(current_parsing_index+1, s1)
			parse_tree[i].insert(current_parsing_index+2, s2)
			
			# INSERTING RULE APPLIED at current_parsing index
			rules_applied[i].insert(rule_index,[0,ysplit])
			
			print("APPLIED RULE 3, with split position: ",ysplit)
			
			for x in range(s1[1],s1[1]+s1[3]):
				for y in range(s1[2],s1[2]+s1[4]):
					images[i,x,y]=s1[0]
			for x in range(s2[1],s2[1]+s2[3]):
				for y in range(s2[2],s2[2]+s2[4]):
					images[i,x,y]=s2[0]  
					
		if state[0]==2:
			
			# Select one of the 4 rules applicable for floors. 
			chosen_one = npy.random.randint(1,high=5)
			print("THE CHOSEN ONE:",chosen_one)
			
			if chosen_one==1:
				s1 = copy.deepcopy(state)
				s1[0]=1
				
#                 parse_tree[i][current_parsing_index][0]=1
				parse_tree[i].insert(current_parsing_index+1,s1) 
				rules_applied[i].insert(rule_index,[1,-1])
				
				for x in range(state[1],state[1]+state[3]):
					for y in range(state[2],state[2]+state[4]):
						images[i,x,y]=1
				
				print("APPLIED RULE 1.")
			if chosen_one==2:
				# Must choose a vertical split position
#                 xsplit = npy.random.randint(state[1]+1,high=state[1]+state[3]) 

				if state[3]==1:
					xsplit=0
				else:                     
					xsplit = npy.random.randint(1,high=state[3])
				s1 = [1, state[1], state[2], xsplit, state[4]]
				s2 = [2, state[1]+xsplit, state[2], state[3]-xsplit, state[4]]
				
				parse_tree[i].insert(current_parsing_index+1, s1)
				parse_tree[i].insert(current_parsing_index+2, s2)
				
				rules_applied[i].insert(rule_index,[2,xsplit])
				
				for x in range(s1[1],s1[1]+s1[3]):
					for y in range(s1[2],s1[2]+s1[4]):
						images[i,x,y]=s1[0]
				for x in range(s2[1],s2[1]+s2[3]):
					for y in range(s2[2],s2[2]+s2[4]):
						images[i,x,y]=s2[0]
						
				print("APPLIED RULE 2, with split:",xsplit)
				
			if chosen_one==3:
				# Must choose a vertical split position
#                 xsplit = npy.random.randint(state[1]+1,high=state[1]+state[3]) 
				if state[3]==1:
					xsplit=0
				else:                     
					xsplit = npy.random.randint(1,high=state[3])
				
				s1 = [0, state[1], state[2], xsplit, state[4]]
				s2 = [2, state[1]+xsplit, state[2], state[3]-xsplit, state[4]]
				
				parse_tree[i].insert(current_parsing_index+1, s1)
				parse_tree[i].insert(current_parsing_index+2, s2)
				
				rules_applied[i].insert(rule_index,[3,xsplit])
				
				for x in range(s1[1],s1[1]+s1[3]):
					for y in range(s1[2],s1[2]+s1[4]):
						images[i,x,y]=s1[0]
				for x in range(s2[1],s2[1]+s2[3]):
					for y in range(s2[2],s2[2]+s2[4]):
						images[i,x,y]=s2[0] 
						
				print("APPLIED RULE 3, with split:",xsplit)
			
			if chosen_one==4:
			   
				s1 = copy.deepcopy(state)
				s1[0]=0
				
				parse_tree[i].insert(current_parsing_index+1,s1) 
				rules_applied[i].insert(rule_index,[4,-1])
				
				for x in range(state[1],state[1]+state[3]):
					for y in range(state[2],state[2]+state[4]):
						images[i,x,y]=0
						
				print("APPLIED RULE 4.")
			
		current_parsing_index+=1
		rule_index+=1
		print("Image:",npy.transpose(images[i]))
		print("Parse Tree:", parse_tree[i])


# In[7]:

# Should initialize images created from these grammars. 
for i in range(num_images):
	
	# Set the state as the entire image: First 5 are state, next 2 are indices
	# State: [c,x,y,W,H]
	state = [3,0,0,10,10]
	parse_tree[i].append(state) 
	current_parsing_index = 0
	rule_index = 0
	# While there are 2's and 3's in the images. 
	
	print("Parsing Image: ",i)
	while ((images[i]==2).any() or (images[i]==3).any()):
		
		
		state = parse_tree[i][current_parsing_index]

		  
		# If it's a facade, split into Floor and Facade from y to y+H             
		if state[0]==3:
			
			if state[4]==1:
				ysplit=-1
				s1 = copy.deepcopy(state)
				s1[0]=2
				parse_tree[i].insert(current_parsing_index+1,s1)
			else: 
				ysplit = npy.random.randint(1,high=state[4]) #Add this to the state index 
#                 ysplit = npy.random.random_integers(1,high=state[4]) #Add this to the state index 
			# First a floor
				s1 = [2,state[1],state[2],state[3],ysplit]
				# The remainder is a facade, don't change label
				s2 = [3,state[1],state[2]+ysplit,state[3],state[4]-ysplit]

	#             parse_tree[i][current_parsing_index][1][0] = index_to_insert
	#             parse_tree[i][current_parsing_index][1][1] = index_to_insert+1
	#             index_to_insert+=2

				# MUST NOW INSERT s1 and s2 INTO THE TREE JUST AFTER CURRENT PARSING INDEX
				parse_tree[i].insert(current_parsing_index+1, s1)
				parse_tree[i].insert(current_parsing_index+2, s2)

			# INSERTING RULE APPLIED at current_parsing index
				for x in range(s2[1],s2[1]+s2[3]):
					for y in range(s2[2],s2[2]+s2[4]):
						images[i,x,y]=s2[0]  

			rules_applied[i].insert(rule_index,[0,ysplit])                    

			for x in range(s1[1],s1[1]+s1[3]):
				for y in range(s1[2],s1[2]+s1[4]):
					images[i,x,y]=s1[0]
					
		if state[0]==2:
			
			# Select one of the 4 rules applicable for floors. 
			chosen_one = npy.random.randint(1,high=5)
#             print("THE CHOSEN ONE:",chosen_one)
			
			if chosen_one==1:
				parse_tree[i][current_parsing_index][0]=1
				rules_applied[i].insert(rule_index,[1,-1])
				
				for x in range(state[1],state[1]+state[3]):
					for y in range(state[2],state[2]+state[4]):
						images[i,x,y]=1
				
#                 print("APPLIED RULE 1.")
			if chosen_one==2:
				# Must choose a vertical split position
#                 xsplit = npy.random.randint(state[1]+1,high=state[1]+state[3]) 

				if state[3]==1:
					xsplit=0
				else:                     
					xsplit = npy.random.randint(1,high=state[3])
				s1 = [1, state[1], state[2], xsplit, state[4]]
				s2 = [2, state[1]+xsplit, state[2], state[3]-xsplit, state[4]]
				
				parse_tree[i].insert(current_parsing_index+1, s1)
				parse_tree[i].insert(current_parsing_index+2, s2)
				
				rules_applied[i].insert(rule_index,[2,xsplit])
				
				for x in range(s1[1],s1[1]+s1[3]):
					for y in range(s1[2],s1[2]+s1[4]):
						images[i,x,y]=s1[0]
				for x in range(s2[1],s2[1]+s2[3]):
					for y in range(s2[2],s2[2]+s2[4]):
						images[i,x,y]=s2[0]
						
#                 print("APPLIED RULE 2, with split:",xsplit)
				
			if chosen_one==3:
				# Must choose a vertical split position
#                 xsplit = npy.random.randint(state[1]+1,high=state[1]+state[3]) 
				if state[3]==1:
					xsplit=0
				else:                     
					xsplit = npy.random.randint(1,high=state[3])
				
				s1 = [0, state[1], state[2], xsplit, state[4]]
				s2 = [2, state[1]+xsplit, state[2], state[3]-xsplit, state[4]]
				
				parse_tree[i].insert(current_parsing_index+1, s1)
				parse_tree[i].insert(current_parsing_index+2, s2)
				
				rules_applied[i].insert(rule_index,[3,xsplit])
				
				for x in range(s1[1],s1[1]+s1[3]):
					for y in range(s1[2],s1[2]+s1[4]):
						images[i,x,y]=s1[0]
				for x in range(s2[1],s2[1]+s2[3]):
					for y in range(s2[2],s2[2]+s2[4]):
						images[i,x,y]=s2[0] 
						
#                 print("APPLIED RULE 3, with split:",xsplit)
			
			if chosen_one==4:
				parse_tree[i][current_parsing_index][0]=0
				rules_applied[i].insert(rule_index,[4,-1])
				
				for x in range(state[1],state[1]+state[3]):
					for y in range(state[2],state[2]+state[4]):
						images[i,x,y]=0
						
#                 print("APPLIED RULE 4.")
			
		current_parsing_index+=1
		rule_index+=1
#         print("Image:",npy.transpose(images[i]))
#         print("Parse Tree:", parse_tree[i])


# In[8]:

for i in range(50):
	plt.imshow(images[i])
	plt.show()


# In[9]:

noisy_images = images.copy()
noise = 0.5*npy.random.rand(200,10,10)

noisy_images += noise
for i in range(num_images):
	noisy_images[i]/=noisy_images[i].max()


# In[10]:

for i in range(30):
	plt.imshow(noisy_images[i])
	plt.show()


# In[ ]:




# In[47]:

# # Saving everything
# npy.save("Parse_Tree.npy",parse_tree)
# npy.save("Rules_Applied.npy",rules_applied)
# npy.save("Images.npy",images)
# npy.save("Noisy_Images.npy",noisy_images)


# In[ ]:

# Defining function to return reward
def return_reward(image_index, parsed_image)
	return (abs(images[image_index]-parsed_image)).sum()

# Partial image reward:
def return_partial_reward(image_index, state)    
	return (abs(images[image_index,state[1]:state[1]+state[3],state[2]:state[2]+state[4]]-state[0])).sum()


# In[ ]:

# Starting on Q Learning: 
# The state space is a tabular form of: (Label, X, Y, W, H)
qvalues = npy.zeros((4,10,10,10,10))



# In[ ]:

# Should initialize images created from these grammars. 
for i in range(num_images):
	
	# Set the state as the entire image: First 5 are state, next 2 are indices
	# State: [c,x,y,W,H]
	state = [3,0,0,10,10]
	parse_tree[i].append(state) 
	current_parsing_index = 0
	rule_index = 0
	# While there are 2's and 3's in the images. 
	
	print("Parsing Image: ",i)
	while ((images[i]==2).any() or (images[i]==3).any()):
				
		state = parse_tree[i][current_parsing_index]
		  
		# If it's a facade, split into Floor and Facade from y to y+H             
		if state[0]==3:
			
			if state[4]==1:
				ysplit=-1
				s1 = copy.deepcopy(state)
				s1[0]=2
				parse_tree[i].insert(current_parsing_index+1,s1)
			else: 
				ysplit = npy.random.randint(1,high=state[4]) #Add this to the state index 
#                 ysplit = npy.random.random_integers(1,high=state[4]) #Add this to the state index 
			# First a floor
				s1 = [2,state[1],state[2],state[3],ysplit]
				# The remainder is a facade, don't change label
				s2 = [3,state[1],state[2]+ysplit,state[3],state[4]-ysplit]

	#             parse_tree[i][current_parsing_index][1][0] = index_to_insert
	#             parse_tree[i][current_parsing_index][1][1] = index_to_insert+1
	#             index_to_insert+=2

				# MUST NOW INSERT s1 and s2 INTO THE TREE JUST AFTER CURRENT PARSING INDEX
				parse_tree[i].insert(current_parsing_index+1, s1)
				parse_tree[i].insert(current_parsing_index+2, s2)

			# INSERTING RULE APPLIED at current_parsing index
				for x in range(s2[1],s2[1]+s2[3]):
					for y in range(s2[2],s2[2]+s2[4]):
						images[i,x,y]=s2[0]  

			rules_applied[i].insert(rule_index,[0,ysplit])                    

			for x in range(s1[1],s1[1]+s1[3]):
				for y in range(s1[2],s1[2]+s1[4]):
					images[i,x,y]=s1[0]
					
		if state[0]==2:
			
			# Select one of the 4 rules applicable for floors. 
			chosen_one = npy.random.randint(1,high=5)
#             print("THE CHOSEN ONE:",chosen_one)
			
			if chosen_one==1:
				parse_tree[i][current_parsing_index][0]=1
				rules_applied[i].insert(rule_index,[1,-1])
				
				for x in range(state[1],state[1]+state[3]):
					for y in range(state[2],state[2]+state[4]):
						images[i,x,y]=1
				
#                 print("APPLIED RULE 1.")
			if chosen_one==2:
				# Must choose a vertical split position
#                 xsplit = npy.random.randint(state[1]+1,high=state[1]+state[3]) 

				if state[3]==1:
					xsplit=0
				else:                     
					xsplit = npy.random.randint(1,high=state[3])
				s1 = [1, state[1], state[2], xsplit, state[4]]
				s2 = [2, state[1]+xsplit, state[2], state[3]-xsplit, state[4]]
				
				parse_tree[i].insert(current_parsing_index+1, s1)
				parse_tree[i].insert(current_parsing_index+2, s2)
				
				rules_applied[i].insert(rule_index,[2,xsplit])
				
				for x in range(s1[1],s1[1]+s1[3]):
					for y in range(s1[2],s1[2]+s1[4]):
						images[i,x,y]=s1[0]
				for x in range(s2[1],s2[1]+s2[3]):
					for y in range(s2[2],s2[2]+s2[4]):
						images[i,x,y]=s2[0]
						
#                 print("APPLIED RULE 2, with split:",xsplit)
				
			if chosen_one==3:
				# Must choose a vertical split position
#                 xsplit = npy.random.randint(state[1]+1,high=state[1]+state[3]) 
				if state[3]==1:
					xsplit=0
				else:                     
					xsplit = npy.random.randint(1,high=state[3])
				
				s1 = [0, state[1], state[2], xsplit, state[4]]
				s2 = [2, state[1]+xsplit, state[2], state[3]-xsplit, state[4]]
				
				parse_tree[i].insert(current_parsing_index+1, s1)
				parse_tree[i].insert(current_parsing_index+2, s2)
				
				rules_applied[i].insert(rule_index,[3,xsplit])
				
				for x in range(s1[1],s1[1]+s1[3]):
					for y in range(s1[2],s1[2]+s1[4]):
						images[i,x,y]=s1[0]
				for x in range(s2[1],s2[1]+s2[3]):
					for y in range(s2[2],s2[2]+s2[4]):
						images[i,x,y]=s2[0] 
						
#                 print("APPLIED RULE 3, with split:",xsplit)
			
			if chosen_one==4:
				parse_tree[i][current_parsing_index][0]=0
				rules_applied[i].insert(rule_index,[4,-1])
				
				for x in range(state[1],state[1]+state[3]):
					for y in range(state[2],state[2]+state[4]):
						images[i,x,y]=0
						
#                 print("APPLIED RULE 4.")
			
		current_parsing_index+=1
		rule_index+=1
#         print("Image:",npy.transpose(images[i]))
#         print("Parse Tree:", parse_tree[i])

