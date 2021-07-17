from tensorflow.keras import layers
import tensorflow as tf

class MaxPoolWithArgMax(layers.Layer):
    def __init__(self, stride=2, **kwargs):
        super(MaxPoolWithArgMax, self).__init__(**kwargs)

        # self.stride = stride

    def call(self, inputs, stride=2):
        _, mask = tf.nn.max_pool_with_argmax(inputs, ksize=[1, stride, 1, 1], strides=[1, stride,1, 1], padding='SAME')
        # Stop the back propagation of the mask gradient calculation
        mask = tf.stop_gradient(mask) # bc of argmax mask has shape [old_shape[0],old_shape[1], old_shape[2]/stride, old_shape[3]]
        # Calculating the maximum pooling operation
        net = tf.nn.max_pool(inputs, ksize=[1, stride, 1, 1], strides=[1, stride, 1, 1], padding='SAME')
        # Return the pooling result and the mask
        return net, mask
