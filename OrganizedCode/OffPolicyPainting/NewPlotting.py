#!/usr/bin/env python
from headers import *

class PlotManager():

	def __init__(self, to_plot=False,data_loader=None):
		
		# self.parser = parser
		self.data_loader = data_loader
		self.plot = to_plot
		self.image_size = self.data_loader.image_size
		image_index = 0
		
		self.fig, self.ax = plt.subplots(1,3,sharey=True)
		if self.plot:			
			self.fig.show()

			self.pred_labels = npy.zeros((self.data_loader.image_size,self.data_loader.image_size))			
			self.sc1 = self.ax[0].imshow(self.pred_labels,aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			self.sc1.set_clim([-1,1])
			self.ax[0].set_title("Predicted Labels")
			self.ax[0].set_adjustable('box-forced')
	 
			self.sc2 = self.ax[1].imshow(self.data_loader.labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			self.sc2.set_clim([-1,1])
			self.ax[1].set_title("Parse Tree")
			self.ax[1].set_adjustable('box-forced')

			if self.data_loader.num_channels==3:
				print("HELLO")
				img = cv2.cvtColor(self.data_loader.images[image_index],cv2.COLOR_RGB2BGR)
				self.sc3 = self.ax[2].imshow(img,aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			else:
				self.sc3 = self.ax[2].imshow(self.data_loader.images[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			# self.sc3.set_clim([self.data_loader.images.min(),self.data_loader.images.max()])
			self.ax[2].set_title("Actual Image")
			self.ax[2].set_adjustable('box-forced')

			# self.sc4 = self.ax[3].imshow(self.data_loader.labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
			# # self.sc4 = self.ax[3].imshow(self.true_labels[image_index],aspect='equal',cmap='jet') #, extent=[0,self.image_size,0,self.image_size],origin='lower')
			# self.sc4.set_clim([-1,1])
			# self.ax[3].set_title("Segmented Painted Image")
			# self.ax[3].set_adjustable('box-forced')			

			self.fig.canvas.draw()

			if self.plot:
				plt.pause(1)	

	def parse_tree_plotting(self):
		self.mask = -npy.ones((self.data_loader.image_size,self.data_loader.image_size))
		self.display_discount = 0.8
		self.backward_discount = 0.98

		for j in range(self.current_parsing_index):
			self.dummy_state = self.current_parse_tree[j]
			self.mask[self.dummy_state.x:self.dummy_state.x+self.dummy_state.w,self.dummy_state.y:self.dummy_state.y+self.dummy_state.h] = -(self.backward_discount**j)
		
		for j in range(self.current_parsing_index,len(self.current_parse_tree)):
			self.dummy_state = self.current_parse_tree[j]
			self.mask[self.dummy_state.x:self.dummy_state.x+self.dummy_state.w,self.dummy_state.y:self.dummy_state.y+self.dummy_state.h] = (self.display_discount**(j-self.current_parsing_index))

	def update_plot_data(self, image_index, pred_labels, parse_tree, parsing_index):
	
		self.current_parse_tree = copy.deepcopy(parse_tree)
		self.current_parsing_index = copy.deepcopy(parsing_index)
		self.pred_labels = pred_labels
		self.alternate_predicted_labels = npy.zeros((self.data_loader.image_size,self.data_loader.image_size))
		self.alternate_predicted_labels[npy.where(self.pred_labels==1)]=1.
		self.alternate_predicted_labels[npy.where(self.pred_labels==2)]=-1.

		self.fig.suptitle("Processing Image: {0}".format(image_index)) 
		self.sc1.set_data(self.alternate_predicted_labels)
		self.parse_tree_plotting()
		self.sc2.set_data(self.mask)
		self.sc3.set_data(self.data_loader.images[image_index])

		# Plotting split line segments from the parse tree.
		split_segs = []
		for j in range(len(self.current_parse_tree)):

			colors = ['r']

			if self.current_parse_tree[j].label==0:

				# Horizontal
				if (self.current_parse_tree[j].rule_applied==0):
					sc = self.current_parse_tree[j].boundaryscaled_split
					split_segs.append([[self.current_parse_tree[j].y,self.current_parse_tree[j].x+sc],[self.current_parse_tree[j].y+self.current_parse_tree[j].h,self.current_parse_tree[j].x+sc]])
					
				# Vertical
				if (self.current_parse_tree[j].rule_applied==1):
					sc = self.current_parse_tree[j].boundaryscaled_split
					split_segs.append([[self.current_parse_tree[j].y+sc,self.current_parse_tree[j].x],[self.current_parse_tree[j].y+sc,self.current_parse_tree[j].x+self.current_parse_tree[j].w]])

		split_lines = LineCollection(split_segs, colors='k', linewidths=2)
		split_lines2 = LineCollection(split_segs, colors='k',linewidths=2)
		split_lines3 = LineCollection(split_segs, colors='k',linewidths=2)

		self.split_lines = self.ax[1].add_collection(split_lines)				
		self.split_lines2 = self.ax[2].add_collection(split_lines2)
		self.split_lines3 = self.ax[0].add_collection(split_lines3)

		# if len(self.parser.start_list)>0 and len(self.parser.goal_list)>0:
		# 	segs = [[npy.array([0,0]),self.parser.start_list[0]]]
		# 	color_index = ['k']
		# 	linewidths = [1]

		# 	for i in range(len(self.parser.goal_list)-1):
		# 		segs.append([self.parser.start_list[i],self.parser.goal_list[i]])
		# 		# Paint
		# 		color_index.append('y')
		# 		linewidths.append(5)
		# 		segs.append([self.parser.goal_list[i],self.parser.start_list[i+1]])
		# 		# Don't paint.
		# 		color_index.append('k')
		# 		linewidths.append(1)

		# 	# Add final segment.
		# 	segs.append([self.parser.start_list[-1],self.parser.goal_list[-1]])
		# 	color_index.append('y')
		# 	linewidths.append(5)

		# 	lines = LineCollection(segs, colors=color_index,linewidths=linewidths)
		# 	self.lines = self.ax[0].add_collection(lines)	
		
		self.fig.canvas.draw()
		# raw_input("Press any key to continue.")
		# self.fig.savefig
		# self.fig.savefig("Image_{0}_Step_{1}.png".format(image_index,self.current_parsing_index),format='png',bbox_inches='tight')
		plt.pause(0.1)	
		# plt.pause(0.5)	

		# if len(self.ax[0].collections):
		# 	del self.ax[0].collections[-1]
	
		del self.ax[0].collections[-1]
		del self.ax[2].collections[-1]			
		del self.ax[1].collections[-1]
