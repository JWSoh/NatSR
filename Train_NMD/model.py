from utils import *
from ops import *

class NMD(object):
    def __init__(self, input):
        self.input=input
        self.build_model()

    def build_model(self, reuse=False):
        print('Build Model NMD')
        with tf.variable_scope("CLASSIFIER", reuse=reuse):

            self.conv1_1 = conv2d(self.input, 64, [3, 3], scope='conv1_1', activation='ReLU')
            self.conv1_2 = conv2d(self.conv1_1, 64, [3, 3],scope='conv1_2', activation='ReLU')
            self.pool1=maxpool(self.conv1_2)

            self.conv2_1 = conv2d(self.pool1, 128, [3, 3], scope='conv2_1', activation='ReLU')
            self.conv2_2 = conv2d(self.conv2_1, 128, [3, 3], scope='conv2_2', activation='ReLU')
            self.pool2 = maxpool(self.conv2_2)

            self.conv3_1 = conv2d(self.pool2, 256, [3, 3], scope='conv3_1', activation='ReLU')
            self.conv3_2 = conv2d(self.conv3_1, 256, [3, 3], scope='conv3_2', activation='ReLU')
            self.pool3 = maxpool(self.conv3_2)

            self.conv4_1 = conv2d(self.pool3, 512, [3, 3], scope='conv4_1', activation='ReLU')
            self.conv4_2 = conv2d(self.conv4_1, 512, [3, 3], scope='conv4_2', activation='ReLU')
            self.pool4 = maxpool(self.conv4_2)

            self.conv5_1 = conv2d(self.pool4, 1, [3, 3], scope='conv5_1')

            self.logit = tf.reduce_mean(self.conv5_1,axis=(1,2))

            self.output = sigmoid(self.logit)