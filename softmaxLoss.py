import os
import sys
sys.path.insert(0,'/data/wczhang/open_tools/caffe-quant/python')
import numpy as np
import caffe


class softmaxWithLossLayer(caffe.Layer):
	"""
	bottom[0]:predict
	bottom[1]:label
 
	https://blog.csdn.net/zxj942405301/article/details/72723056
	"""
	
	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception("need two inputs.")
			
	def reshape(self, bottom, top):
		if bottom[0].num != bottom[1].num:
			raise Exception("input predict and gt must have the same dimension.")
        
		self.prob = np.zeros_like(bottom[0].data, dtype = np.float32)
		top[0].reshape(1)

	def forward(self, bottom, top):
		top[0].data[0] = 0.
		norm = 0.

		for i in xrange(bottom[0].num):
			mmax = np.max(bottom[0].data[i])
			bottom[0].data[i] -= mmax
			norm = np.sum(np.exp(bottom[0].data[i]))
			
			self.prob[i, ...] = np.exp(bottom[0].data[i]) / norm #softmax function
			gt_label = bottom[1].data[i]
			prob_label = np.exp(bottom[0].data[i]) / norm
			gt_idx = int(gt_label)

			top[0].data[0] -= np.sum(np.log(prob_label[gt_idx]))
		top[0].data[...] /= bottom[0].num
		
	def backward(self, top, propagate_down, bottom):
		if not propagate_down[0]:
			return
			
		self.diff = np.zeros_like(bottom[0].data, dtype = np.float32)
		for i in xrange(0, bottom[0].num):
			gt_label = int(bottom[1].data[i])
			for j in xrange(0, len(bottom[0].data[0])):
				if j == gt_label:
					self.diff[i,j] = self.prob[i,j] - 1  
				else:
					self.diff[i,j] = self.prob[i,j]
 
		bottom[0].diff[...] = self.diff / bottom[0].num
