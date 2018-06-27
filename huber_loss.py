import os
import sys
caffe_root = '/data/wczhang/open_tools/caffe-quant'
sys.path.insert(0, caffe_root + '/python')
import numpy as np
import caffe

class HuberLossLayer(caffe.Layer):

	def setup(self, bottom, top):
		if len(bottom)!= 2:
			raise Exception("need two inputs to compute distance.")
			 
	def reshape(self, bottom, top):
		if bottom[0].count != bottom[1].count:
			raise Exception("input must have the same dimension.")
		self.diff = np.zeros_like(bottom[0].data, dtype = np.float32)
		top[0].reshape(1)
		
	def forward(self, bottom, top):

          	tempb0 = np.reshape(bottom[0].data,self.diff.shape)
          	tempb1 = np.reshape(bottom[1].data,self.diff.shape)
		#print "bottom[0].data:",bottom[0].data.shape  ##bottom[0].data: (50, 10)
		#print "bottom[1].data:",bottom[1].data.shape ##bottom[1].data: (50, 1, 1, 10)
    		#print "self.diff.shape:", self.diff.shape   ##self.diff.shape: (50, 10)	
		           
          	self.diff[...] = tempb0 - tempb1

      		self.sigm = 1.5
      		idx = np.abs(self.diff) <= self.sigm
      		idx1 = np.abs(self.diff) > self.sigm
      		top[0].data[...] = (np.sum((self.diff[idx]**2) * 0.5) + np.sum(self.sigm * np.abs(self.diff[idx1]) - 0.5 * self.sigm)) / bottom[0].num
		
	def backward(self, top, propagate_down, bottom):
      		idx = np.abs(self.diff) <= self.sigm
      		idx1 = self.diff > self.sigm
      		idx2 = self.diff < -1.0 * self.sigm

      		for i in range(0,2):
        		if not propagate_down[i]:
          			continue
        		if i == 0:
          			sign = 1
        		else:
          			sign = -1
          
        		#print "idx:",idx
        		#print  "self.diff[idx]:",self.diff[idx]
        		#print  "bottom[i].diff[idx]:",bottom[0].diff[idx]
        		bottom[i].diff[idx] = sign * self.diff[idx] / bottom[i].num
        		bottom[i].diff[idx1] = sign * self.sigm / bottom[i].num
        		bottom[i].diff[idx2] = sign * (-1.0 * self.sigm) / bottom[i].num
