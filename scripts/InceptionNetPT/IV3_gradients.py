#!/usr/bin/env python 
import numpy as npy
import os
import random
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras
import cv2
import sys
import h5py
import argparse

class GradientNet():

	# def __init__(self, sess, base_filepath=''):
	def __init__(self, sess):

		# Defining Inception V3 Architecture pre-trained on imagenet. 
		self.image_size = (256,256,3)
		# self.batch_size = 32
		self.base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=self.image_size)
		self.image_mean = npy.array([ 175.5183833 ,  176.6830765 ,  192.35719172])

		# # Inception V3 for us has 2 outputs; adding 2 dense layers for this output.
		x = self.base_model.output
		x = keras.layers.GlobalAveragePooling2D()(x)
		x = keras.layers.Dense(1024,activation='relu')(x)		
		# Modifying to predict 256 values instead of 2.
		# self.class_predictions = keras.layers.Dense(self.image_size[0],activation='softmax')(x)
		self.pred_grads = keras.layers.Dense(self.image_size[0],activation='softmax')(x)
		# Compiling the model.
		self.model = keras.models.Model(inputs=self.base_model.input, outputs=self.pred_grads)
		
		adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		self.model.compile(optimizer=adam,loss='categorical_crossentropy')
		# self.model.compile(optimizer=adam,loss='kld')
		# self.base_filepath = base_filepath
		self.num_images = 276
		self.num_epochs = 50
		# Just because so few images.
		self.batch_size = 12

	def preprocess(self):
		self.images = self.images.astype(float)
		self.images -= self.image_mean
		self.image_gradients = npy.zeros((self.num_images,2,self.image_size[0]))
		for i in range(self.num_images):
			self.image_gradients[i,0,:-1] = self.gradients[i][0]
			self.image_gradients[i,1,:-1] = self.gradients[i][1]

	def save_weights(self,k):
		self.model.save_weights("model_epoch{0}.h5".format(k))

	def load_model_weights(self, model_file, weight_file):

		# Load the model.
		with open(model_file,"r") as f:
			self.model = keras.models.model_from_yaml(f.read())

		# Load the weights:
		self.model.load_weights(weight_file)
		adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		self.model.compile(optimizer=adam,loss='categorical_crossentropy')

	# def define_crossentropy(self, true_gradients, predicted_gradients):



	def train(self):
		
		self.batch_inputs = npy.zeros((self.batch_size,self.image_size[0],self.image_size[1],self.image_size[2]))
		self.batch_targets = npy.zeros((self.batch_size,self.image_size[0]))

		e = 0	
		print("########################################")
		print("Processing Epoch:",e)

		for e in range(1,self.num_epochs+1):
			print("########################################")
			print("Processing Epoch:",e)

			index_list = range(self.num_images)
			npy.random.shuffle(index_list)
			
			for i in range(self.num_images/self.batch_size):

				indices = index_list[i*self.batch_size:(i+1)*self.batch_size]
				self.batch_inputs = self.images[[indices]]
				# Picking up HORIZONTAL GRADIENTS now
				self.batch_targets = self.image_gradients[[indices],0].reshape((self.batch_size,self.image_size[0]))

				# Train the model on this batch.
				self.model.fit(self.batch_inputs,self.batch_targets)

			self.save_weights(e)
			self.forward(e)

	# def evaluator(self):
	def forward(self, epoch):

		index_list = range(self.num_images)
		self.predicted_gradients = npy.zeros((self.num_images,self.image_size[0]))

		for i in range(self.num_images/self.batch_size):
			indices = index_list[i*self.batch_size:(i+1)*self.batch_size]
			self.batch_inputs = self.images[[indices]]
			
			self.predicted_gradients[[indices],:] = self.model.predict_on_batch(self.batch_inputs)

		npy.save("Predicted_gradients_{0}.npy".format(epoch),self.predicted_gradients)

def parse_arguments():

	parser = argparse.ArgumentParser(description='Primitive-Aware Segmentation Argument Parsing')
	parser.add_argument('--images',dest='images',type=str)
	parser.add_argument('--gradients',dest='gradients',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()

	# # Create a TensorFlow session with limits on GPU usage.
	gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list='0,1')
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


