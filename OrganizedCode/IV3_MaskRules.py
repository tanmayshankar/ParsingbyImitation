#!/usr/bin/env python
from headers import *
import Model_Class
import Parsing
import Plotting_Utilities

def MetaClass():

	def __init__(self, arguments=None):

		self.args = arguments

		# Instantiate Model Class.
		self.model = Model_Class.Model()
		if self.args.pretrain:
			hierarchical_model.create_modular_net(sess,load_pretrained_mod=True,base_model_file=self.args.base_model,pretrained_weight_file=self.args.model)
		else:
			hierarchical_model.create_modular_net(sess,load_pretrained_mod=False,base_model_file=self.args.base_model)

		# Instantiate parser class.
		self.parser = Parsing.Parser(model_instance=self.model)

		self.plot_manager = Plotting_Utilities.PlotManager(to_plot=self.args.plot,parser=self.parser)


	# Checked this - should be good - 11/1/18
	def meta_training(self,train=True):
		
		image_index = 0
		self.painted_image = -npy.ones((self.image_size,self.image_size))
		self.predicted_labels = npy.zeros((self.num_images,self.image_size, self.image_size))
		self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))
		# self.minimum_width = self.paintwidth
		
		if self.plot:			
			self.define_plots()

		self.to_train = train		
		if not(train):
			self.num_epochs=1
			self.annealed_epsilon=self.final_epsilon

		print("Entering Training Loops.")
		# For all epochs
		for e in range(self.num_epochs):

			self.save_model_weights(e)				
			for i in range(self.num_images):

				print("Initializing Tree.")
				self.initialize_tree()
				print("Starting Parse Tree.")
				self.construct_parse_tree(i)
				print("Propagating Rewards.")
				self.propagate_rewards()				

				print("#___________________________________________________________________________")
				print("Epoch:",e,"Training Image:",i,"TOTAL REWARD:",self.parse_tree[0].reward)
	
				if train:
					self.backprop(i)
					if e<self.decay_epochs:
						epsilon_index = e*self.num_images+i
						self.annealed_epsilon = self.initial_epsilon-epsilon_index*self.annealing_rate
					else: 
						self.annealed_epsilon = self.final_epsilon
				# embed()
				self.start_list = []
				self.goal_list = []
							
			if train:
				npy.save("parsed_{0}.npy".format(e),self.predicted_labels)
				npy.save("painted_images_{0}.npy".format(e),self.painted_images)

				if ((e%self.save_every)==0):
					self.save_model_weights(e)				
			else: 
				npy.save("validation_{0}.npy".format(self.suffix),self.predicted_labels)
				npy.save("validation_painted_{0}.npy".format(self.suffix))
				
			self.predicted_labels = npy.zeros((self.num_images,self.image_size,self.image_size))
			self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))

	def map_rules_to_state_labels(self, rule_index):
		if (rule_index<=3):
			return [0,0]
		if (rule_index==4):
			return 1
		if (rule_index==5):
			return 2

	def preprocess(self):
		for i in range(self.num_images):
			self.images[i] = cv2.cvtColor(self.images[i],cv2.COLOR_RGB2BGR)
		
		self.images = self.images.astype(float)
		self.images -= self.images.mean(axis=(0,1,2))
		
def parse_arguments():
	parser = argparse.ArgumentParser(description='Primitive-Aware Segmentation Argument Parsing')
	parser.add_argument('--images',dest='images',type=str)
	parser.add_argument('--labels',dest='labels',type=str)
	parser.add_argument('--size',dest='size',type=int)
	parser.add_argument('--minwidth',dest='minimum_width',type=int)
	parser.add_argument('--base_model',dest='base_model',type=str)
	parser.add_argument('--suffix',dest='suffix',type=str)
	parser.add_argument('--gpu',dest='gpu')
	parser.add_argument('--plot',dest='plot',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--pretrain',dest='pretrain',type=int,default=0)
	parser.add_argument('--model',dest='model',type=str)
	parser.add_argument('--parse_steps',dest='steps',type=int,default=9)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	print(args)

	# # Create a TensorFlow session with limits on GPU usage.
	gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list=args.gpu)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	keras.backend.tensorflow_backend.set_session(sess)
	keras.backend.set_session(sess)
	keras.backend.set_learning_phase(1)

	# with sess.as_default():
	hierarchical_model = MetaClass()


	print("Loading Images.")
	hierarchical_model.images = npy.load(args.images)
	print("Loading Labels.")
	hierarchical_model.true_labels = npy.load(args.labels) 

	hierarchical_model.image_size = args.size 
	print("Preprocessing Images.")
	hierarchical_model.preprocess()

	hierarchical_model.minimum_width = args.minimum_width
	hierarchical_model.max_parse_steps = args.steps
	hierarchical_model.plot = args.plot
	hierarchical_model.to_train = args.train
	
	if hierarchical_model.to_train:
		hierarchical_model.suffix = args.suffix
	print("Starting to Train.")

	hierarchical_model.meta_training(train=args.train)









if __name__ == '__main__':
	main(sys.argv)
