from ops import *

class FRACTAL(object):
    def __init__(self, x, scale,reuse=False):
        self.input=x
        self.scale=scale
        self.reuse=reuse

        self.build_model(reuse=self.reuse)

    def build_model(self, reuse=False):
        print('Build Model GEN')
        with tf.variable_scope('GEN', reuse=reuse):
            self.conv1 = conv2d(self.input, 64, [3, 3], scope='conv1', activation=None)

            self.head = self.conv1

            for idx in range(0,2):
                self.head = self.dense_block(self.head, 6, 'Block' + repr(idx))
            self.conv2=tf.add(self.head,self.conv1)

            self.head=self.conv2
            for idx in range(2,4):
                self.head = self.dense_block(self.head, 6, 'Block' + repr(idx))
            self.conv3=tf.add(self.head,self.conv2)
            self.conv3=tf.add(self.conv3, self.conv1)

            self.head=self.conv3
            for idx in range(4,6):
                self.head = self.dense_block(self.head, 6, 'Block' + repr(idx))
            self.conv4=tf.add(self.head,self.conv3)

            self.head=self.conv4
            for idx in range(6,8):
                self.head = self.dense_block(self.head, 6, 'Block' + repr(idx))
            self.conv5=tf.add(self.head,self.conv4)
            self.conv5=tf.add(self.conv5,self.conv3)
            self.conv5=tf.add(self.conv5, self.conv1)

            self.head=self.conv5
            self.out1 = conv2d(self.head, 64, [3, 3], scope='conv2', activation=None)
            self.out2 = tf.add(self.conv1, self.out1)

            if self.scale==4:
                self.conv_up1 = conv2d(self.out2, 64 * self.scale // 2 * self.scale //2 , [3, 3], scope='conv_up1',
                                  activation=None)
                self.conv2_1 = tf.depth_to_space(self.conv_up1, self.scale // 2)

                self.conv_up2 = conv2d(self.conv2_1, 64 * self.scale// 2 * self.scale//2, [3, 3], scope='conv_up2',
                                      activation=None)
                self.conv2_2 = tf.depth_to_space(self.conv_up2, self.scale // 2)
            else:
                self.conv_up1 = conv2d(self.out2, 64 * self.scale * self.scale, [3, 3], scope='conv_up1',
                                       activation=None)
                self.conv2_2 = tf.depth_to_space(self.conv_up1, self.scale)

            self.output = conv2d(self.conv2_2, 3, [3, 3], scope='conv_out', activation=None)

        tf.add_to_collection('InNOut', self.input)
        tf.add_to_collection('InNOut', self.output)

    def dense_block(self, input_x, nb_layers, scope):
        with tf.variable_scope(scope):
            layers_concat = []
            layers_concat.append(input_x)

            for i in range(nb_layers - 1):
                x = tf.concat(layers_concat,axis=-1)
                x = conv2d(x, 64, [3,3], scope='conv'+str(i), activation='ReLU')
                layers_concat.append(x)

            x = tf.concat(layers_concat, axis=-1)
            x = conv2d(x, 64,[1,1], scope='conv_fusion')

            return tf.add(input_x,0.1*x)


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

            self.out = sigmoid(self.logit)

class Discriminator(object):
    def __init__(self, input, reuse=False):
        self.input = input
        self.reuse=reuse
        self.build_model()

    def build_model(self):
        print('Build Model DIS')
        with tf.variable_scope("DIS", reuse=self.reuse) as scope:
            if self.reuse:
                scope.reuse_variables()

            self.conv1_1 = SNconv(self.input, 64, [3, 3], scope='conv1_1', activation='leakyReLU')
            self.conv1_2 = SNconv(self.conv1_1, 64, [3, 3],strides=2, scope='conv1_2', activation='leakyReLU')

            self.conv2_1 = SNconv(self.conv1_2, 128, [3, 3], scope='conv2_1', activation='leakyReLU')
            self.conv2_2 = SNconv(self.conv2_1, 128, [3, 3], strides=2, scope='conv2_2', activation='leakyReLU')

            self.conv3_1 = SNconv(self.conv2_2, 256, [3, 3], scope='conv3_1', activation='leakyReLU')
            self.conv3_2 = SNconv(self.conv3_1, 256, [3, 3],strides=2, scope='conv3_2', activation='leakyReLU')

            self.conv4_1 = SNconv(self.conv3_2, 512, [3, 3], scope='conv4_1', activation='leakyReLU')
            self.conv4_2 = SNconv(self.conv4_1, 512, [3, 3],strides=2, scope='conv4_2', activation='leakyReLU')

            self.conv5_1 = SNconv(self.conv4_2, 1024, [3, 3], scope='conv5_1', activation='leakyReLU')
            self.conv5_2 = SNconv(self.conv5_1, 1024, [3, 3], strides=2, scope='conv5_2', activation='leakyReLU')

            self.conv6_1 = SNconv(self.conv5_2, 1, [3, 3], scope='conv6_1', activation=None)

            self.logit = tf.reduce_mean(self.conv6_1, axis=(1,2))
            self.out = tf.nn.sigmoid(self.logit)
