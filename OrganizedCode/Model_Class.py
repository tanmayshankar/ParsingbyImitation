#!/usr/bin/env python
from headers import *

class Model():

	def __init__(self):
		self.image_size = 256

	def load_base_model(self, sess, model_file=None):

		# Define a KERAS / tensorflow session. 
		self.sess = sess

		# We are always loading the model from the the gradient file at the least. 
		self.base_model = keras.models.load_model(model_file)
		
		# The common FC layers are defined here: Input features are keras layer outputs of base model. 
		for layers in self.base_model.layers:
			if layers.name=='fc6_features':
				self.fc6_features = layers.output
			# Switching to presoftmax values. This is because we are masking then normalizing. 
			if layers.name=='vertical_presf_grads':
				self.vertical_split_probs = layers.output
				# self.vertical_split_probs = layers
			if layers.name=='horizontal_presf_grads':
				self.horizontal_split_probs = layers.output
				# self.horizontal_split_probs = layers

	def define_rule_stream(self):
		# Now defining rule FC:
		self.rule_num_branches = 4
		self.target_rule_shapes = 6
		self.rule_num_fclayers = 2
		self.rule_num_hidden = 256

		self.rule_fc = keras.layers.Dense(self.rule_num_hidden,activation='relu')(self.fc6_features)
		self.premask_rule_probabilities = keras.layers.Dense(self.target_rule_shapes,activation='softmax',name='rule_probabilities')(self.rule_fc)
		self.rule_mask = keras.layers.Input(batch_shape=(1,self.target_rule_shapes),name='rule_mask')

		self.masked_unnorm_rule_probs = keras.layers.Multiply()([self.premask_rule_probabilities,self.rule_mask])
		self.masked_rule_sum = keras.backend.sum(self.masked_unnorm_rule_probs)
		self.masked_norm_rule_probs = keras.layers.Lambda(lambda x,y: tf.divide(x,y), arguments={'y': self.masked_rule_sum}, name='masked_norm_rule_probs')(self.masked_unnorm_rule_probs)

		self.rule_loss_weight = keras.backend.variable(0.,name='rule_loss_weight')

	def define_split_stream(self):

		self.split_loss_weight = [keras.backend.variable(0.,name='split_loss_weight{0}'.format(j)) for j in range(2)]
		self.split_mask = keras.layers.Input(batch_shape=(1,self.image_size-1),name='split_mask')

		self.masked_unnorm_horizontal_probs = keras.layers.Multiply()([self.horizontal_split_probs,self.split_mask])
		self.masked_unnorm_vertical_probs = keras.layers.Multiply()([self.vertical_split_probs,self.split_mask])
		
		self.masked_hgrad_sum = keras.backend.sum(self.masked_unnorm_horizontal_probs)		
		self.masked_vgrad_sum = keras.backend.sum(self.masked_unnorm_vertical_probs)
		
		self.masked_norm_horizontal_probs = keras.layers.Lambda(lambda x,y: tf.divide(x,y), arguments={'y': self.masked_hgrad_sum},name='masked_horizontal_probabilities')(self.masked_unnorm_horizontal_probs)
		self.masked_norm_vertical_probs = keras.layers.Lambda(lambda x,y: tf.divide(x,y), arguments={'y': self.masked_vgrad_sum},name='masked_vertical_probabilities')(self.masked_unnorm_vertical_probs)

	def define_primitive_stream(self):
		# Defining primitive FC layers.
		self.primitive_num_hidden = 256
		self.num_primitives = 4

		self.primitive_fc0 = keras.layers.Dense(self.primitive_num_hidden,activation='relu')(self.fc6_features)
		self.primitive_probabilities = keras.layers.Dense(self.num_primitives,activation='softmax',name='primitive_probabilities')(self.primitive_fc0)		

		self.primitive_targets = keras.backend.placeholder(shape=(self.num_primitives),name='primitive_targets')
		self.primitive_loss_weight = keras.backend.variable(0.,name='primitive_loss_weight')

	def define_keras_model(self):
		############################################################################################### 
	
		self.keras_model = keras.models.Model(inputs=[self.base_model.input,self.split_mask,self.rule_mask],
										outputs=[self.masked_norm_rule_probs,
												 self.masked_norm_horizontal_probs,
												 self.masked_norm_vertical_probs])		

		print("Model successfully defined.")
			
		# Defining optimizer.
		self.adam_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=10.)

		# # Option to freeze base model layers.
		# for layer in self.base_model.layers:
		# 	layer.trainable = False
			
		self.keras_model.compile(optimizer=self.adam_optimizer,loss='categorical_crossentropy',loss_weights={'masked_norm_rule_probs': self.rule_loss_weight,
																									   'masked_horizontal_probabilities': self.split_loss_weight[0],
																									   'masked_vertical_probabilities': self.split_loss_weight[1]})

		print("Supposed to have compiled model.")
		# embed()
		# Because of how this is defined, we must call a Callback in Fit. 

	def save_model_weights(self,k):
		self.keras_model.save_weights("model_weights_epoch{0}.h5".format(k))

	def save_model(self,k):
		self.keras_model.save("model_file_epoch{0}.h5".format(k))

	def load_pretrained_model(self, model_file):
		# Load the model - instead of defining from scratch.
		self.keras_model = keras.models.load_model(model_file)
		
	def load_model_weights(self,weight_file):
		self.keras_model.load_weights(weight_file)

	def create_modular_net(self, sess, load_pretrained_mod=False, base_model_file=None, pretrained_weight_file=None):

		print("Training Policy from base model.")
		self.load_base_model(sess, base_model_file)
		self.define_rule_stream()
		self.define_split_stream()
		self.define_keras_model()
		if load_pretrained_mod:
			print("Now loading pretrained model.")
			self.load_model_weights(pretrained_weight_file)	