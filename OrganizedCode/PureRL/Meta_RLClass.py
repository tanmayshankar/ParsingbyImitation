#!/usr/bin/env python
from headers import *
import TF_Model_Class
import Data_Loader

class Meta_RLClass():

	def __init__(self, session=None,arguments=None):

		self.sess = session
		self.args = arguments
		self.batch_size = 10
		self.num_epochs = 100

		# Instantiate Model Class.
		self.model = TF_Model_Class.Model()

		if self.args.model:
			self.model.create_network(self.sess,pretrained_weight_file=self.args.model,to_train=self.args.train)
		else:
			self.model.create_network(self.sess,to_train=self.args.train)

		# Instantiate data loader class to load and preprocess the data.
		self.data_loader = Data_Loader.DataLoader(image_path=self.args.images,label_path=self.args.labels,indices_path=self.args.indices,rewards_path=self.args.horrew)
		self.data_loader.preprocess()

		# self.plot_manager = Plotting_Utilities.PlotManager(to_plot=self.args.plot,parser=self.parser)

	def forward(self, indices):
		# Forward pass of the network.		

		num_images_batch = len(indices)
		indices = npy.array(indices)

		redo_indices = npy.ones((num_images_batch)).astype(bool)

		self.batch_samples = npy.zeros((num_images_batch))

		while redo_indices.any():
			self.batch_samples[redo_indices] = self.sess.run(self.model.sample_split,
				feed_dict={self.model.input: self.data_loader.images[indices[redo_indices]].reshape(num_images_batch, self.model.image_size, self.model.image_size, 1) })[:,0]
			redo_indices = (self.batch_samples<0.)+(self.batch_samples>1.)

		# We are rescaling the image to 1 to 255.
		# Putting this in a different vector because we need to backprop with the original.
		self.rescaled_batch_samples = (self.batch_samples*(self.model.image_size-1)+1.).astype(int)
		for j in range(num_images_batch):
			self.painted_images[indices[j],:self.rescaled_batch_samples[j],:] = 1.
		# self.painted_images[indices,:self.rescaled_batch_samples[range(num_images_batch)],:] = 1.

	def compute_reward(self, indices):
		# Normalizing the rewards.
		reward_values = (self.painted_images[indices]*self.data_loader.labels[indices]).mean(axis=(1,2))		
		self.split_return_weight_vect = reward_values / self.data_loader.horizontal_rewards[indices]

		self.rewards[indices] = self.split_return_weight_vect

	def backprop(self, indices):		
		# Updating the network.
		self.sess.run(self.model.train, feed_dict={self.model.input: self.data_loader.images[indices],
												   self.model.sampled_split: self.batch_samples,
												   self.model.split_return_weight: self.split_return_weight_vect})

	def meta_training(self,train=True):
		
		image_index = 0
		self.painted_images = -npy.ones((self.data_loader.num_images, self.model.image_size,self.model.image_size))
		self.rewards = npy.zeros((self.data_loader.num_images))

		if self.args.plot:		
			self.define_plots()

		if not(self.args.train):
			self.num_epochs=1

		print("Entering Training Loops.")
		for e in range(self.num_epochs):

			self.model.save_model(e)
	
			index_list = range(self.data_loader.num_images)
			npy.random.shuffle(index_list)

			for i in range(self.data_loader.num_images/self.batch_size):
	
				indices = index_list[i*self.batch_size:min(self.data_loader.num_images,(i+1)*self.batch_size)]				
				
				# Forward pass over these images.
				self.forward(indices)
				self.compute_reward(indices)

				print("#__________________________________________")
				print("Epoch:",e,"Training Batch:",i)				

				# Backward pass if we are training.				
 				if self.args.train:
					self.backprop(indices)
			
			print("AFTER EPOCH:",e,"AVERAGE REWARD:",self.rewards.mean())		
			if self.args.train:
				npy.save("painted_images_{0}.npy".format(e),self.painted_images)
				npy.save("rewards_{0}.npy".format(e),self.rewards)
				if ((e%self.save_every)==0):
					self.save_model_weights(e)				
			else: 
				npy.save("validation_{0}.npy".format(self.suffix),self.painted_images)
				npy.save("val_rewards.npy".format(e),self.rewards)

			self.painted_images = -npy.ones((self.data_loader.num_images, self.model.image_size,self.model.image_size))
			self.rewards = npy.zeros((self.data_loader.num_images))	
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
	config = tf.ConfigProto(gpu_options=gpu_ops,log_device_placement=True)
	sess = tf.Session(config=config)

	hierarchical_model = Meta_RLClass(session=sess,arguments=args)

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
