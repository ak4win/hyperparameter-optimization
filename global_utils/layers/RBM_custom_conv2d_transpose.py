from tensorflow.keras import layers
import tensorflow.keras.backend as K

# import tensorflow as tf
# import numpy as np


class CustomConv2DTranspose(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        output_shape,
        activation,
        initial_weights,
        initial_biases,
        **kwargs
    ):
        super(CustomConv2DTranspose, self).__init__(**kwargs)

        self.initial_weights = initial_weights
        self.initial_biases = initial_biases

        self.filters = filters
        self.kernel_size = kernel_size
        self.output_shape_ = output_shape
        self.activation = {"tanh": K.tanh, "relu": K.relu, "sigmoid": K.sigmoid}[
            activation
        ]

        self.trainable = True

    def build(self, _):
        self.w = self.initial_weights
        self.b = self.initial_biases

    def call(self, inputs, **kwargs):
        strides = (self.kernel_size, 1)
        convoluted_output = K.conv2d_transpose(
            inputs, self.w, self.output_shape_, strides=strides, padding="same"
        )

        outputs = convoluted_output + self.b
        activations = self.activation(outputs)
        return activations

        # Stack overflow
        # outputs = K.conv2d_transpose(inputs, self.kernel, self.output_shape_)
        # outputs = K.bias_add(outputs, self.bias)
        # self.shape = outputs.shape
        # outputs = K.reshape(outputs, [-1, self.shape[1] * self.shape[2] * self.shape[3]])
        # return outputs

    def get_weights(self):
        return [self.w, self.b]

    def compute_output_shape(self, input_shape):
        return self.output_shape_
