import tensorflow as tf

def conv2d(x,filters,kernel, strides=1,dilation=1, scope=None, activation=None, reuse=None):
    if activation is None:
        with tf.variable_scope(scope):
            out = tf.layers.conv2d(x, filters, kernel, strides=strides, padding='SAME', dilation_rate=dilation,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2d',
                                   reuse=reuse)
        return out

    elif activation == 'ReLU':
        with tf.variable_scope(scope):
            out = tf.layers.conv2d(x, filters, kernel, strides=strides, padding='SAME', dilation_rate=dilation,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name='conv2d',
                                   reuse=reuse)
        return tf.nn.relu(out)

    elif activation == 'leakyReLU':
        with tf.variable_scope(scope):
            out = tf.layers.conv2d(x, filters, kernel, strides=strides, padding='SAME', dilation_rate=dilation,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2d',
                                   reuse=reuse)
        return tf.nn.leaky_relu(out,0.2)

def bn(x, is_train):
    return tf.layers.batch_normalization(x, training=is_train)

def relu(x):
    return tf.nn.relu(x)

def lrelu(x):
    return tf.nn.leaky_relu(x, 0.2)

def maxpool(x):
    return tf.layers.max_pooling2d(x,[2,2],[2,2],padding='valid')

def fc(x, num_h, scope):
    with tf.variable_scope(scope):
        out= tf.layers.dense(x,num_h, name='dense')
    return out

def sigmoid(x):
    return tf.nn.sigmoid(x)