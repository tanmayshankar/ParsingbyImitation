#!/usr/bin/env python
from headers import *
from state_class import *

class ModularNet():

	def __init__(self):

		self.num_epochs = 20
		self.save_every = 100
		self.num_images = 276
		self.current_parsing_index = 0
		self.parse_tree = [parse_tree_node()]
		self.paintwidth = -1
		self.minimum_width = -1
		self.images = []
		self.original_images = []
		self.true_labels = []
		self.image_size = -1
		self.intermittent_lambda = 0.
		self.suffix = []

		# For Epsilon Greedy Policy: 
		self.initial_epsilon = 0.9
		self.final_epsilon = 0.1
		self.decay_epochs = 5
		self.annealing_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_epochs*self.num_images)
		self.annealed_epsilon = 0.

	def initialize_keras_model(self, sess, model_file=None):

		# Define a KERAS / tensorflow session. 
		self.sess = sess

		# We are always loading the model from the the gradient file at the least. 
		self.base_model = keras.models.load_model(model_file)

		# Input features are keras layer outputs of base model. 
		for layers in self.base_model.layers:
			if layers.name=='fc6_features':
				self.fc6_features = layers.output
			if layers.name=='vertical_grads':
				self.vertical_grads = layers.output
			if layers.name=='horizontal_grads':
				self.horizontal_grads = layers.output

		###############################################################################################
		# The common FC layers are defined above. 

		###############################################################################################
		# Now defining rule FC:
		self.rule_num_branches = 4
		self.target_rule_shapes = [6,4,4,2]
		self.rule_num_fclayers = 2
		self.rule_num_hidden = 256

		self.rule_fc = [[[] for i in range(self.rule_num_fclayers)] for j in range(self.rule_num_branches)]
		
		for j in range(self.rule_num_branches):
			self.rule_fc[j][0] = keras.layers.Dense(self.rule_num_hidden,activation='relu')(self.fc6_features)
			self.rule_fc[j][1] = keras.layers.Dense(self.target_rule_shapes[j],activation='softmax')(self.rule_fc[j][0])

		# self.selected_rule_probabilities = tf.case({tf.equal(self.rule_indicator,0): lambda: self.rule_probabilities[0], tf.equal(self.rule_indicator,1): lambda: self.rule_probabilities[1], 
		# 							tf.equal(self.rule_indicator,2): lambda: self.rule_probabilities[2], tf.equal(self.rule_indicator,3): lambda: self.rule_probabilities[3]},default=lambda: -tf.zeros(1),exclusive=True,name='selected_rule_probabilities')

		self.selected_rule_probabilities = tf.case({tf.equal(self.rule_indicator,0): lambda: self.rule_fc[0][1], tf.equal(self.rule_indicator,1): lambda: self.rule_fc[1][1], 
									tf.equal(self.rule_indicator,2): lambda: self.rule_fc[2][1], tf.equal(self.rule_indicator,3): lambda: self.rule_fc[3][1]},default=lambda: -tf.zeros(1),exclusive=True,name='selected_rule_probabilities')

def parse_arguments():

	parser = argparse.ArgumentParser(description='Primitive-Aware Segmentation Argument Parsing')
	parser.add_argument('--images',dest='images',type=str)
	parser.add_argument('--gradients',dest='gradients',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()

	# # Create a TensorFlow session with limits on GPU usage.
	gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list='2,3')
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)
	
	KTF.set_session(sess)

	gradnet = GradientNet(sess)

	gradnet.images = npy.load(args.images)
	gradnet.gradients = npy.load(args.gradients)
	gradnet.preprocess()

	gradnet.train()

if __name__ == '__main__':
	main(sys.argv)

