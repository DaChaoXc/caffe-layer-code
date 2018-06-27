import os
import sys
sys.path.insert(0,'/data/wczhang/open_tools/caffe-quant/python')
import numpy as np
import caffe

"""
Focal Loss for face classify
paper: Focal Loss for Dense Object Detection
"""

class FocalLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        # bottom[0]:fc layer out
		# bottom[0].count=100
		# bottom[0].num=50(batch size)
		# bottom[0].channel=2(binary classify)

		# bottom[1]:label
		# bottom[1]num=50
		# bottom[1].count=50
		# bottom[1].channel=1(label)

        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
			
        if bottom[0].channels != 2:
            raise Exception("only support binary classification")

        if bottom[0].num != bottom[1].num:
            raise Exception("Input pre and classify label must have the same dimension.")
			
        self.alpha = 0.25
        self.gamma = 2.0

  	
    def reshape(self, bottom, top):
        self.diff = np.zeros_like(bottom[0].data, dtype = np.float32)
        self.prob = np.zeros_like(bottom[0].data, dtype = np.float32)#fc layer size: 2, one img has two probs
        top[0].reshape(1)

    def softmax(self, x1, x2):
        return np.exp(x1)/(np.exp(x1)+np.exp(x2))
	
    def forward(self, bottom, top):
        top[0].data[...] = 0
        #print 'bottom[1].data',bottom[1].data
        #print 'bottom[1].data[49]', bottom[1].data[49]
        #print bottom[0].data

        for i in xrange(0, bottom[0].num):#start from 0~bottom[0].num-1s
            #print 'i=',i
            gt_label = int(bottom[1].data[i])
            #print 'bottom[0].data[i,0]',bottom[0].data[i,0]
            #print 'bottom[0].data[i,1]', bottom[0].data[i,1]
            prob = self.softmax(bottom[0].data[i,1], bottom[0].data[i,0])#attention
            if gt_label == 1:
                pt = prob#pos
                #print 'gt_label == 1, pt=',pt
            else:
                pt = 1 - prob#neg
                #print 'gt_label == 0, pt=',pt

            self.prob[i,:] = [1 - prob, prob]
            top[0].data[...] -= self.alpha * (1 - pt)**self.gamma * np.log(pt)

        #print 'self.prob',self.prob
        top[0].data[...] /= bottom[0].num

    def backward(self, top, propagate_down, bottom):
        if not propagate_down[0]:
            return

        self.diff = 0.
        for i in xrange(0, bottom[0].num):
            gt_label = int(bottom[1].data[i])
            p0 = self.prob[i,0]
            p1 = self.prob[i,1]

            if gt_label == 0:
                self.diff[i,0] = -1*(1 - p0)**self.gamma * (1 - p0 - self.gamma * p0 * np.log(p0))
                self.diff[i,1] = p1 * (1 - p0)**(self.gamma - 1) * (1 - p0 - self.gamma * p0 * np.log(p0))
            elif gt_label == 1:
                self.diff[i,0] = p0 * (1 - p1)**(self.gamma - 1) * (1 - p1 - self.gamma * p1 * np.log(p1))
                self.diff[i,1] = -1*(1 - p1)**self.gamma * (1 - p1 - self.gamma * p1 * np.log(p1))
        loss_weight = top[0].diff[0]
        bottom[0].diff[...] = loss_weight * self.diff / bottom[0].num