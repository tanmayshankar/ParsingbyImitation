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

		# Maintaining list of all goals and start locations. 
		self.goal_list = []
		self.start_list = []
		self.previous_goal = npy.zeros(2)
		self.current_start = npy.zeros(2)

	def load_base_model(self, sess, model_file=None):

		# Define a KERAS / tensorflow session. 
		self.sess = sess

		# We are always loading the model from the the gradient file at the least. 
		self.base_model = keras.models.load_model(model_file)
		
		# The common FC layers are defined here: Input features are keras layer outputs of base model. 
		for layers in self.base_model.layers:
			if layers.name=='fc6_features':
				self.fc6_features = layers.output
			if layers.name=='vertical_grads':
				self.vertical_grads = layers.output
			if layers.name=='horizontal_grads':
				self.horizontal_grads = layers.output

	def define_rule_stream(self):
		# Now defining rule FC:
		self.rule_num_branches = 4
		self.target_rule_shapes = [6,4,4,2]
		self.rule_num_fclayers = 2
		self.rule_num_hidden = 256

		self.rule_fc = [[[] for i in range(self.rule_num_fclayers)] for j in range(self.rule_num_branches)]
		
		for j in range(self.rule_num_branches):
			self.rule_fc[j][0] = keras.layers.Dense(self.rule_num_hidden,activation='relu')(self.fc6_features)
			self.rule_fc[j][1] = keras.layers.Dense(self.target_rule_shapes[j],activation='softmax')(self.rule_fc[j][0])

		# Maintaining the rule indicators as Keras input layers, rather than just variables. 
		self.rule_indicator = keras.layers.Input(batch_shape=(1,),dtype='int32',name='rule_indicator')
		
		# Selecting which rule probability distribution to sample from.
		self.lambda_rule_probs = keras.layers.Lambda(lambda x: keras.backend.control_flow_ops.case({tf.equal(x,0)[0]: lambda: self.rule_fc[0][1],
																									tf.equal(x,1)[0]: lambda: self.rule_fc[1][1],
																									tf.equal(x,2)[0]: lambda: self.rule_fc[2][1],
																									tf.equal(x,3)[0]: lambda: self.rule_fc[3][1]},
																									default=lambda: -tf.zeros(1),exclusive=True,name='selected_rule_probabilities'))(self.rule_indicator)
		#No need to explicitly define losses for each branch. 
		#Providing targets and the selected lambda rule probabilities should take care of this. 

	def define_split_stream(self):
		# Split FC values are already defined; now creating Split distributions.

		self.split_indicator = keras.layers.Input(batch_shape=(1,),dtype='int32',name='split_indicator')

		self.horizontal_split_dist = tf.contrib.distributions.Categorical(probs=self.horizontal_grads,name='horizontal_split_dist')
		self.vertical_split_dist = tf.contrib.distributions.Categorical(probs=self.vertical_grads,name='vertical_split_dist')
	
		# Providing split location probabilities rather than the sampled split location, 
		# because with Categorical distribution of splits, can now do epsilon greedy sampling. 
		# With Gaussian distributions, we didn't need this because we could control variance.
		self.lambda_split_probs = keras.layers.Lambda(lambda x: keras.backend.control_flow_ops.case({tf.equal(x,0)[0]: lambda: self.horizontal_grads,
																									 tf.equal(x,1)[0]: lambda: self.vertical_grads},
																									 default=lambda: -tf.ones(1),exclusive=True,name='selected_split_probabilities' ) )(self.split_indicator)
		#No need to explicitly define losses for each branch. 
		#Providing targets and the selected lambda split probabilities should take care of this. 

	def define_primitive_stream(self):
		# Defining primitive FC layers.
		self.primitive_num_hidden = 256
		self.num_primitives = 4
		self.primitive_fc0 = keras.layers.Dense(self.primitive_num_hidden,activation='relu')(self.fc6_features)
		self.primitive_probabilities = keras.layers.Dense(self.num_primitives,activation='softmax',name='primitive_probabilities')(self.primitive_fc0)		
		self.primitive_dist = tf.contrib.distributions.Categorical(probs=self.primitive_probabilities,name='primitive_dist')
		self.sampled_primitive = self.primitive_dist.sample()

	def define_keras_model(self):
		############################################################################################### 

		# Defining the new model.
		self.model = keras.models.Model(inputs=[self.base_model.input,self.rule_indicator,self.split_indicator],
										outputs=[self.lambda_rule_probs, self.lambda_split_probs, self.primitive_probabilities])
		print("Model successfully defined.")
			
		adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	
		# Option to freeze base model layers.
		# for layer in self.base_model.layers:
		# 	layer.trainable = False
		
		# Compiling the new model
		self.model.compile(optimizer=adam,loss={'selected_rule_probabilities': 'categorical_crossentropy',
												'selected_split_probabilities': 'categorical_crossentropy',
												'primitive_probabilities': 'categorical_crossentropy'})

	def create_modular_net(self, sess, model_file=None):
		self.load_base_model(sess, model_file)
		self.define_rule_stream()
		self.define_split_stream()
		self.define_primitive_stream()
		self.define_keras_model()

###JUST FOR TRIAL:

gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list="1,2")
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

mn = ModularNet()
mn.create_modular_net(sess,"T2_named/model_file_epoch374.h5")
embed()	