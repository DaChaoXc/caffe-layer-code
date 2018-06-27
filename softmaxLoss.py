import os
import sys
caffe_root = ''
sys.path.insert(0, caffe_root + '/python')
import numpy as np
import caffe

class softmaxWithLoss(caffe.Layer):
	"""
	bottom[0]:predict
	bottom[1]:label
 
  	https://blog.csdn.net/zxj942405301/article/details/72723056
	"""
	
	def setup(self, bottom, top):
  		if len(bottom) != 2:
  			  raise Exception("need two inputs.")
			
	def reshape(self, bottom, top):
  		if bottom[0].count != bottom[1].count:
  			  raise Exception("input predict and gt must have the same dimension.")
        
  		self.prob = np.zeros_like(bottom[0].data, dtype = np.float32)
  		top[0].reshape(1)

	def forward(self, bottom, top):
  		top[0].data[0] = 0.
      		sum = 0.
      
  		for i in xrange(bottom[0].num):#deal with every sample
  			  sum = np.sum(np.exp(bottom[0].data[i])
        
  			  self.prob[i, ...] = np.exp(bottom[0].data[i]) / sum #softmax function
  			  top[0].data[0] -= np.sum(bottom[1].data[i] * np.log(np.exp(bottom[0].data[i]) / sum))
        
  		top[0].data[...] /= bottom[0].num
   
	def backward(self, top, propagate_down, bottom):
      		if not propagate_down[i]:
          		continue	
        	
      		self.diff = 0.
      
      		for i in xrange(0, bottom[0].num):
          		gt_label = int(bottom[1].data[i])
          		for j in xrange(0, bottom[0].channel):
              			if j == gt_label:
                  			self.diff[i,j] = self.prob[i,j] - 1  
              			else:
                  			self.diff[i,j] = self.prob[i,j]
                  
      		bottom[0].diff[...] = self.diff / bottom[0].num
