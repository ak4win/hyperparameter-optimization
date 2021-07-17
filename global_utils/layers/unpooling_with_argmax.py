from tensorflow.keras import layers
import tensorflow as tf

class UnMaxPoolWithArgmax(layers.Layer):
    def __init__(self, stride=2, **kwargs):
        super(UnMaxPoolWithArgmax, self).__init__(**kwargs)

        self.stride = stride

    def call(self, inputs, mask):
        ksize = [1, self.stride,1, 1]
        input_shape = inputs.get_shape().as_list()
        #  calculation new shape - scale by the stride of the pooling layer
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
        b = one_like_mask * batch_range # bc of broadcasting this will have shape [batch_size, 1, mask_size, 1]
        y = mask // (output_shape[2] * output_shape[3]) # (0 or 1) // 240 > 0 - y has the same shape as mask
        x = mask % (output_shape[2] * output_shape[3]) // output_shape[3] # 120 % 240 // 1 = 120
        feature_range = tf.range(output_shape[3], dtype=tf.int64) # range = 1 - due to one dimensional data
        f = one_like_mask * feature_range # as feature_range is 1, this doesnt change anythong
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(inputs)
        stacked = tf.stack([b, y, x, f])
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size])) # (tf.stack([[1,1,120,1], y, x, f]) >> output - 120 x 4
        values = tf.reshape(inputs, [updates_size]) # Flatten
        ret = tf.scatter_nd(indices, values, output_shape) # Put [values] at [indices] of an empty tensor with shape [shape]
        return ret
