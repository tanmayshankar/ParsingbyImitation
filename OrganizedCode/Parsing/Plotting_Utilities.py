#!/usr/bin/env python
from headers import *

class PlotManager():

	def __init__(self, to_plot=False, parser=None):
		
		self.parser = parser
		self.plot = to_plot
		image_index = 0

		self.fig, self.ax = plt.subplots(2,4,sharey=True)
		if self.plot:
			self.fig.show()
		
		self.sc1 = self.ax[0,0].imshow(self.parser.predicted_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
		self.sc1.set_clim([-1,1])
		self.ax[0,0].set_title("Predicted Labels")
		self.ax[0,0].set_adjustable('box-forced')

		self.sc2 = self.ax[0,1].imshow(self.parser.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
		self.sc2.set_clim([-1,1])
		self.ax[0,1].set_title("Parse Tree")
		self.ax[0,1].set_adjustable('box-forced')

		self.sc3 = self.ax[0,2].imshow(self.parser.images[image_index,:,:,0],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
		self.sc3.set_clim([self.parser.images.max(),self.parser.images.min()])
		self.ax[0,2].set_title("Actual Image")
		self.ax[0,2].set_adjustable('box-forced')

		self.sc4 = self.ax[1,1].imshow(self.parser.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
		self.sc4.set_clim([-1,1])
		self.ax[1,1].set_title("Segmented Painted Image")
		self.ax[1,1].set_adjustable('box-forced')			

		self.h_probs = npy.zeros((255))
		self.v_probs = npy.zeros((255))

		self.hprob_manager =  self.ax[0,3].plot(self.h_probs,range(255),'b')
		self.ax[0,3].set_xlim(-0.1,1.1)
		self.ax[0,3].set_adjustable('box-forced')

		self.vprob_manager = self.ax[1,2].plot(range(255),255*self.v_probs,'r')
		self.ax[1,2].set_adjustable('box-forced')

		self.fig.canvas.draw()

		if self.plot:
			plt.pause(0.1)	

	def attention_plots(self):
		self.mask = -npy.ones((self.parser.image_size,self.parser.image_size))
		self.display_discount = 0.8
		self.backward_discount = 0.98

		for j in range(self.parser.current_parsing_index):
			self.dummy_state = self.parser.parse_tree[j]
			self.mask[self.dummy_state.x:self.dummy_state.x+self.dummy_state.w,self.dummy_state.y:self.dummy_state.y+self.dummy_state.h] = -(self.backward_discount**j)
		
		for j in range(self.parser.current_parsing_index,len(self.parser.parse_tree)):
			self.dummy_state = self.parser.parse_tree[j]
			self.mask[self.dummy_state.x:self.dummy_state.x+self.dummy_state.w,self.dummy_state.y:self.dummy_state.y+self.dummy_state.h] = (self.display_discount**(j-self.parser.current_parsing_index))

	def update_plot_data(self, image_index):
	
		# self.alternate_painted_image[npy.where(self.predicted_labels[image_index]==1)]=1.
		self.alternate_painted_image[npy.where(self.parser.painted_images[image_index]==1)]=1.
		self.alternate_predicted_labels[npy.where(self.parser.predicted_labels[image_index]==1)]=1.
		self.alternate_predicted_labels[npy.where(self.parser.predicted_labels[image_index]==2)]=-1.

		self.fig.suptitle("Processing Image: {0}".format(image_index)) 
		self.sc1.set_data(self.alternate_predicted_labels)
		self.attention_plots()
		self.sc2.set_data(self.mask)
		self.sc3.set_data(self.parser.images[image_index])
		self.sc4.set_data(self.alternate_painted_image)

		self.hprob_manager[0].set_data(self.parser.h_probs,range(255))
		self.vprob_manager[0].set_data(range(255),255*self.parser.v_probs)
		# Plotting split line segments from the parse tree.
		split_segs = []
		for j in range(len(self.parser.parse_tree)):

			colors = ['r']

			if self.parser.parse_tree[j].label==0:

				rule_app_map = self.parser.remap_rule_indices(self.parser.parse_tree[j].rule_applied)

				if (self.parser.parse_tree[j].alter_rule_applied==1) or (self.parser.parse_tree[j].alter_rule_applied==3):
					sc = self.parser.parse_tree[j].boundaryscaled_split
					split_segs.append([[self.parser.parse_tree[j].y,self.parser.parse_tree[j].x+sc],[self.parser.parse_tree[j].y+self.parser.parse_tree[j].h,self.parser.parse_tree[j].x+sc]])
					
				if (self.parser.parse_tree[j].alter_rule_applied==0) or (self.parser.parse_tree[j].alter_rule_applied==2):					
					sc = self.parser.parse_tree[j].boundaryscaled_split
					split_segs.append([[self.parser.parse_tree[j].y+sc,self.parser.parse_tree[j].x],[self.parser.parse_tree[j].y+sc,self.parser.parse_tree[j].x+self.parser.parse_tree[j].w]])

		split_lines = LineCollection(split_segs, colors='k', linewidths=2)
		split_lines2 = LineCollection(split_segs, colors='k',linewidths=2)
		split_lines3 = LineCollection(split_segs, colors='k',linewidths=2)

		self.split_lines = self.ax[1].add_collection(split_lines)				
		self.split_lines2 = self.ax[2].add_collection(split_lines2)
		self.split_lines3 = self.ax[3].add_collection(split_lines3)

		if len(self.parser.start_list)>0 and len(self.parser.goal_list)>0:
			segs = [[npy.array([0,0]),self.parser.start_list[0]]]
			color_index = ['k']
			linewidths = [1]

			for i in range(len(self.parser.goal_list)-1):
				segs.append([self.parser.start_list[i],self.parser.goal_list[i]])
				# Paint
				color_index.append('y')
				linewidths.append(5)
				segs.append([self.parser.goal_list[i],self.parser.start_list[i+1]])
				# Don't paint.
				color_index.append('k')
				linewidths.append(1)

			# Add final segment.
			segs.append([self.parser.start_list[-1],self.parser.goal_list[-1]])
			color_index.append('y')
			linewidths.append(5)

			lines = LineCollection(segs, colors=color_index,linewidths=linewidths)
			self.lines = self.ax[0].add_collection(lines)	
		
		self.fig.canvas.draw()
		# raw_input("Press any key to continue.")
		self.fig.savefig
		plt.pause(0.1)	
		# plt.pause(0.5)	

		if len(self.ax[0].collections):
			del self.ax[0].collections[-1]
	
		del self.ax[3].collections[-1]
		del self.ax[2].collections[-1]			
		del self.ax[1].collections[-1]
