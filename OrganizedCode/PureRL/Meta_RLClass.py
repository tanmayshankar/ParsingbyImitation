#!/usr/bin/env python
from headers import *
import Model_Class
import Parsing
import Plotting_Utilities
import Data_Loader

def Meta_RLClass():

	def __init__(self, session=None,arguments=None):

		self.sess = session
		self.args = arguments

		# Instantiate Model Class.
		self.model = Model_Class.Model()

		if self.args.pretrain:
			self.model.create_network(self.sess,load_pretrained_mod=True,base_model_file=self.args.base_model,pretrained_weight_file=self.args.model)
		else:
			self.model.create_network(self.sess,load_pretrained_mod=False,base_model_file=self.args.base_model)

		# Instantiate data loader class to load and preprocess the data.
		self.data_loader = Data_Loader.DataLoader(image_path=self.args.images,label_path=self.args.labels,indices_path=self.args.indices,rewards_path=self.args.horrew)
		self.data_loader.preprocess()

		# self.plot_manager = Plotting_Utilities.PlotManager(to_plot=self.args.plot,parser=self.parser)

	def forward(self):
		# Forward pass of the network.
		

	def compute_reward(self):

	def backward(self):		
		# Updating the network.

	def meta_training(self,train=True):
		
		image_index = 0
		self.painted_image = -npy.ones((self.image_size,self.image_size))
		self.predicted_labels = npy.zeros((self.num_images,self.image_size, self.image_size))
		self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))

		if self.args.plot:		
			self.define_plots()

		if not(self.args.train):
			self.num_epochs=1

		print("Entering Training Loops.")
		for e in range(self.num_epochs):

			self.model.save_model(e)
			
			for i in range(self.num_images):

				# Forward pass over the image.

				print("#___________________________________________________________________________")
				print("Epoch:",e,"Training Image:",i)
		
				# Backward pass if we are training.
 				if self.args.train:
					self.backprop(i)
					
			if self.args.train:
				npy.save("parsed_{0}.npy".format(e),self.predicted_labels)
				npy.save("painted_images_{0}.npy".format(e),self.painted_images)

				if ((e%self.save_every)==0):
					self.save_model_weights(e)				
			else: 
				npy.save("validation_{0}.npy".format(self.suffix),self.predicted_labels)
				npy.save("validation_painted_{0}.npy".format(self.suffix))
				
			self.predicted_labels = npy.zeros((self.num_images,self.image_size,self.image_size))
			self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))
	
def parse_arguments():
	parser = argparse.ArgumentParser(description='Primitive-Aware Segmentation Argument Parsing')
	parser.add_argument('--images',dest='images',type=str)
	parser.add_argument('--labels',dest='labels',type=str)
	parser.add_argument('--indices',dest='indices',type=str)
	parser.add_argument('--horrew',dest='horrew',type=str)
	parser.add_argument('--suffix',dest='suffix',type=str)
	parser.add_argument('--gpu',dest='gpu')
	parser.add_argument('--plot',dest='plot',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	print(args)

	# # Create a TensorFlow session with limits on GPU usage.
	gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list=args.gpu)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	hierarchical_model = MetaClass(session=sess,arguments=args)

	print("Loading Images.")
	hierarchical_model.images = npy.load(args.images)	
	hierarchical_model.true_labels = npy.load(args.labels) 

	hierarchical_model.plot = args.plot
	hierarchical_model.to_train = args.train
	
	if hierarchical_model.to_train:
		hierarchical_model.suffix = args.suffix
	print("Starting to Train.")

	hierarchical_model.meta_training(train=args.train)

if __name__ == '__main__':
	main(sys.argv)
