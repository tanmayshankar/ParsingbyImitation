#!/usr/bin/env python
from headers import *
import ActorCriticTF
import ActorCriticTF_Regularized
import Data_Loader
import DDPGAggrevateParser
import NewPlotting
import Memory

class Meta_RLClass():

	def __init__(self, session=None,arguments=None):

		self.sess = session
		self.args = arguments
		self.batch_size = 5
		self.num_epochs = 250
		self.save_every = 1

		# Instantiate data loader class to load and preprocess the data.
		if self.args.indices:
			self.data_loader = Data_Loader.DataLoader(image_path=self.args.images,label_path=self.args.labels,indices_path=self.args.indices)
		else:
			self.data_loader = Data_Loader.DataLoader(image_path=self.args.images,label_path=self.args.labels)
		self.data_loader.preprocess()
		
		self.args.train = bool(self.args.train)
		# # Instantiate Model Class.		
		self.ActorCriticModel = ActorCriticTF_Regularized.ActorCriticModel(self.sess,to_train=self.args.train)

		if self.args.model:
			self.ActorCriticModel.create_network(self.sess,pretrained_weight_file=self.args.model,to_train=self.args.train)
		else:
			self.ActorCriticModel.create_network(self.sess,to_train=self.args.train)

		# Instantiate memory. 
		self.memory = Memory.Replay_Memory()

		# Instantiate the plotting manager. 
		self.plotting_manager = NewPlotting.PlotManager(to_plot=self.args.plot,data_loader=self.data_loader)		

		# Instantiate parser, passing arguments to take care of train / test / IGM within the parsing code. 
		self.parser = DDPGAggrevateParser.Parser(model_instance=self.ActorCriticModel, data_loader_instance=self.data_loader,
			memory_instance=self.memory, plot_manager=self.plotting_manager, args=self.args, session=self.sess)

	def train(self):
		self.parser.meta_training(self.args.train)

def parse_arguments():
	parser = argparse.ArgumentParser(description='Primitive-Aware Segmentation Argument Parsing')
	parser.add_argument('--images',dest='images',type=str)
	parser.add_argument('--labels',dest='labels',type=str)
	parser.add_argument('--indices',dest='indices',type=str)
	parser.add_argument('--suffix',dest='suffix',type=str)
	parser.add_argument('--plot',dest='plot',type=int,default=0)
	parser.add_argument('--gpu',dest='gpu')
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()

	# # Create a TensorFlow session with limits on GPU usage.
	gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list=args.gpu)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	hierarchical_model = Meta_RLClass(session=sess,arguments=args)
	hierarchical_model.train()

if __name__ == '__main__':
	main(sys.argv)


