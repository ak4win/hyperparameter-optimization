# Set the seed
from numpy.random import seed

seed(1)

# import numpy and tf
import numpy as np
import tensorflow as tf

tf.random.set_seed(2)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [25, 10]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower

# machine learning libaries
from tensorflow.keras import layers
from keras.layers import (
    MaxPooling2D,
    Dense,
    Lambda,
    Flatten,
    Input,
    Reshape,
    UpSampling2D,
    Conv2DTranspose,
)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from keras.losses import mse, mae
import keras.backend as K
from keras.callbacks import ModelCheckpoint

# data science libaries
import pandas as pd

# data preprocessing libaries
from sklearn.preprocessing import MinMaxScaler

# Import own modules
from work.global_utils.get_data_multi_note_without_smoothing import get_data
from work.global_utils.evaluation import per_rms_diff
from work.global_utils.callback import CustomCallback

# own modules layers
from global_utils.layers.custom_conv2d_transpose import CustomConv2DTranspose
from global_utils.layers.max_pooling_with_argmax import MaxPoolWithArgMax
from global_utils.layers.unpooling_with_argmax import UnMaxPoolWithArgmax

# hyper-parameter optimization
from keras_tuner import HyperParameters


def create_model(hp):

    encoder_activation = hp.Choice("encoder_activation", ["relu", "tanh"])
    decoder_activation = hp.Choice("decoder_activation", ["relu", "tanh"])

    # Encoder
    inputs = layers.Input(shape=(sample_size, 1, 1), batch_size=batch_size)

    channels, kernel = 24, 12
    out_conv1 = layers.Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=encoder_activation,
        padding="same",
    )(inputs)
    out_reshape1 = layers.Reshape((-1, 1, 1))(out_conv1)
    out_pool1, mask1 = MaxPoolWithArgMax()(out_reshape1)

    channels, kernel = 12, 9
    out_conv2 = layers.Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=encoder_activation,
        padding="same",
    )(out_pool1)
    out_reshape2 = layers.Reshape((-1, 1, 1))(out_conv2)
    out_pool2, mask2 = MaxPoolWithArgMax()(out_reshape2)

    out_max_pool_1, mask3 = MaxPoolWithArgMax()(out_pool2)

    channels, kernel = 12, 5
    out_conv3 = layers.Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=encoder_activation,
        padding="same",
    )(out_max_pool_1)
    out_reshape3 = layers.Reshape((-1, 1, 1))(out_conv3)
    out_pool3, mask4 = MaxPoolWithArgMax()(out_reshape3)

    channels, kernel = 12, 3
    out_conv4 = layers.Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=encoder_activation,
        padding="same",
    )(out_pool3)
    out_reshape4 = layers.Reshape((-1, 1, 1))(out_conv4)
    out_pool4, mask5 = MaxPoolWithArgMax()(out_reshape4)

    out_max_pool_2, mask6 = MaxPoolWithArgMax()(out_pool4)

    conv_dims = out_max_pool_2.shape[1:]
    out_flatten = Flatten()(out_max_pool_2)
    out_dense1 = Dense(27, activation="relu")(out_flatten)
    z_mean = Dense(5, activation="relu", name="z_mean")(out_dense1)
    z_log_var = Dense(5, activation="relu", name="z_log_var")(out_dense1)
    z = Lambda(sampling, output_shape=(5,), name="z")([z_mean, z_log_var])

    # Decoder
    de_out_dense2 = Dense(27, activation="relu")(z)
    de_out_dense3 = Dense(54, activation="relu")(de_out_dense2)
    de_inverse_flatten = layers.Reshape((-1, 1, 1))(de_out_dense3)

    de_out_pool5 = UnMaxPoolWithArgmax(stride=2)(de_inverse_flatten, mask6)

    channels, kernel = 12, 3
    de_out_pool6 = UnMaxPoolWithArgmax(stride=2)(de_out_pool5, mask5)
    de_out_reshape5 = layers.Reshape(out_conv4.shape[1:])(de_out_pool6)
    de_out_transcov1 = CustomConv2DTranspose(
        channels, kernel, out_pool3.shape, activation=decoder_activation
    )(de_out_reshape5)

    channels, kernel = 12, 5
    de_out_pool7 = UnMaxPoolWithArgmax(stride=2)(de_out_transcov1, mask4)
    de_out_reshape6 = layers.Reshape(out_conv3.shape[1:])(de_out_pool7)
    de_out_transcov2 = CustomConv2DTranspose(
        channels, kernel, out_max_pool_1.shape, activation=decoder_activation
    )(de_out_reshape6)

    de_out_pool8 = UnMaxPoolWithArgmax(stride=2)(de_out_transcov2, mask3)

    channels, kernel = 12, 9
    de_out_pool9 = UnMaxPoolWithArgmax(stride=2)(de_out_pool8, mask2)
    de_out_reshape7 = layers.Reshape(out_conv2.shape[1:])(de_out_pool9)
    de_out_transcov3 = CustomConv2DTranspose(
        channels, kernel, out_pool1.shape, activation=decoder_activation
    )(de_out_reshape7)

    channels, kernel = 24, 12
    de_out_pool10 = UnMaxPoolWithArgmax(stride=2)(de_out_transcov3, mask1)
    de_out_reshape8 = layers.Reshape(out_conv1.shape[1:])(de_out_pool10)
    de_out_transcov4 = CustomConv2DTranspose(
        channels, kernel, inputs.shape, activation=decoder_activation
    )(de_out_reshape8)
    outputs = de_out_transcov4

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
        loss=hp.Choice("loss", hp.Choice("loss", ["mse", "mae"]))
        # metrics=["val_loss"],
    )

    return model
