import os
import sys
sys.path.insert(0,'/data/wczhang/open_tools/caffe-quant/python')
import numpy as np
import caffe

class SmoothLossL1Layer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):
        #tag,need reshape bottom[0] and bottom[1],maybe lmdb don't need
        tempb0 = np.reshape(bottom[0].data,self.diff.shape)
        tempb1 = np.reshape(bottom[1].data,self.diff.shape)
        self.diff = tempb0 - tempb1

        idx = np.abs(self.diff) < 1.0
        idx1 = np.abs(self.diff) >= 1.0

        top[0].data[...] = (np.sum((self.diff[idx]**2) * 0.5) + np.sum(np.abs(self.diff[idx1]) - 0.5)) / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        idx = np.abs(self.diff) < 1.0
        idx1 = self.diff >= 1.0
        idx2 = self.diff <= -1.0

        for i in range(0,2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1

            bottom[i].diff[idx] = sign * self.diff[idx] / bottom[i].num
            bottom[i].diff[idx1] = sign * 1.0 / bottom[i].num
            bottom[i].diff[idx2] = sign * (-1.0) / bottom[i].num