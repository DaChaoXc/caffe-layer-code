import caffe
import numpy as np

class CrossEntropyLoss(caffe.layer):
	"""
	bottom[0]:predict
	bottom[1]:label
	bottom:(N x C x H x W),x∈[−∞,+∞]
			p^=σ(x)∈[0,1],p∈[0,1]
	top:(1 x 1 x 1 x 1)
	σ(x)=1/(1+e^x)
	'Y. zhang, et. al, Deep Mutual Learning (arxiv 1706.00384v1)'
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
		top[0].data[0] = 0;
		for i in xrange(bottom[0].num):#deal with every sample
			norm = np.sum(np.exp(bottom[0].data[i])
			self.prob[i, ...] = np.exp(bottom[0].data[i]) / norm #softmax function			
		
	def backward(self, top, propagate_down, bottom):		
		bottom[0].diff[...]=0
		bottom[1].diff[...]=0
		
		for i in xrange(bottom[0].num):
			for j in xrange(bottom[0].count / bottom[0].num):
				bottom[0].diff[i,j] = self.prob[i,j] - bottom[1].data[i,j]
			bottom[0].diff[...] /= bottom[0].num
	
	
	