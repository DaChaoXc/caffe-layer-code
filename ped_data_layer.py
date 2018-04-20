# coding: utf-8

import sys
caffe_root = '/mnt/vm_share2/caffe'
sys.path.insert(0, caffe_root + '/python')
import caffe
import numpy as np
from PIL import Image
import random
import scipy

class pedDataLoadLayer(caffe.Layer):
	"""
	Directly load (input image, label) pairs for pedestrian classify task	
	Use this to feed data to deep but fast net
	"""
	def setup(self, bottom, top):
		"""
		Setup data layer according to parameters:
		- source: path to ped train and test dir
		- mean: tuple of mean values to subtract
		for Pedestrian Classify.
		"""
		# config
		self.top_names = ['data', 'label']
		params = eval(self.param_str)
		check_params(params)
		self.batch_size = params['batch_size']
		
		# Create a batch loader to load the images.
		self.batch_loader = BatchLoader(params, None)
		
		self.source = params['source']
		self.mean = np.array(params['mean'])
		self.resize_h = params['crop_size'][0]
		self.resize_w = params['crop_size'][1]
		#print self.mean,self.resize_h,self.resize_w
				
		# two tops: data and label
		if len(top) != 2:
			raise Exception("Need to define two tops: data and label.")
		# data layers have no bottoms
		if len(bottom) != 0:
			raise Exception("Do not define a bottom.")
					
		print_info("pedDataLoadLayer", params)		
		
	def reshape(self, bottom, top):
		top[0].reshape(self.batch_size, 1, self.resize_h, self.resize_w)		
		top[1].reshape(self.batch_size, 1) #each image label is a intger
		
	def forward(self, bottom, top):
		for i in range(self.batch_size):
			# Use the batch loader to load the next image.
			im, label = self.batch_loader.load_next_image()

			# Add directly to the caffe data layer
			top[0].data[i, ...] = im
			top[1].data[i, ...] = label
	def backward(self, top, propagate_down, bottom):
		pass
		
class BatchLoader(object):
	"""
	This class abstracts away the loading of images.
	Images can either be loaded singly, or in a batch. The latter is used for
	the asyncronous data layer to preload batches while other processing is performed
	"""
	def __init__(self, params, result):
		self.result = result
		self.source = params['source']
		self.batch_size = params['batch_size']
		self.mean = params['mean']
		self.crop_size = params['crop_size']
		self.isshuffle = params['shuffle']

		# get list of image indexes.
		self.imagelist = open(self.source, 'r').read().splitlines()
		self._cur = 0  # current image

		print "BatchLoader initialized with {} images".format(len(self.imagelist))
	
	def load_next_image(self):
		"""
		Load the next image in a batch.
		"""
		#Did we finish an epoch?
		if self._cur == len(self.imagelist):
			self._cur = 0

			if int(self.isshuffle):
				self._cur = random.randint(0,len(self.imagelist)-1)

		# Load an image
		image_sample = self.imagelist[self._cur]  # Get an image   eg: xxx.jpg 0
		im_path = "/mnt/vm_share2/caffe/data/ped_classify_data/train_ride/" + image_sample.split(' ')[0]
		im = np.asarray(Image.open(im_path))
		
		im = scipy.misc.imresize(im, self.crop_size)  # resize dafault?
		#cv2.resize(self.data, dsize=(resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
		im = (im.astype(np.float32) - self.mean) / 128
		im = np.expand_dims(im, axis=0)
		
		# do a simple horizontal flip as data augmentation
		flip = np.random.choice(2) * 2 - 1 #-1/1
		im = im[:, ::flip, :]

		im = im[:, :, ::-1]  # change to BGR
		#im = im.transpose((2, 0, 1)) #[128 96 48 1]->[128 1 96 48]
		
		# Load and prepare ground truth
		gt_classes=image_sample.split(' ')[1]
		
		self._cur += 1
		return im, int(gt_classes)	
	
	
def check_params(params):
	"""
	A utility function to check the parameters for the data layers.
	"""
	required = ['batch_size', 'source', 'crop_size', 'mean', 'shuffle']
	for r in required:
		assert r in params.keys(), 'Params must include {}'.format(r)	
	
def print_info(name, params):
	"""
	Output some info regarding the class
	"""
	print "{} initialized for with batch_size: {}, crop_size: {}.".format(
		name,params['batch_size'],params['crop_size'])	
	
	
	
	
	
	