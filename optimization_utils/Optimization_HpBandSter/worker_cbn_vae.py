"""
HpBandSter Optimization Downsampling-Convolutional
Restricted Boltzmann Machines Variational Auto-Encoder
============================
CBN-VAE model of our research project is optimized
by the HpBandSter Methdod.

The configuration space shows the types of hyperparameters and
even contains conditional dependencies.

We'll optimise the following hyperparameters:
+-------------------------+----------------+----------------------+------------------------+
| Parameter Name          | Parameter type |  Range/Choices       | Comment                |
+=========================+================+======================+========================+
| Learning rate           |  float         | [1e-6, 1e-2]         | varied logarithmically |
+-------------------------+----------------+----------------------+------------------------+
| Optimizer               | categorical    | {Adam, SGD}          | discrete choice        |
+-------------------------+----------------+----------------------+------------------------+
| SGD momentum            |  float         | [0, 0.99]            | only active if         |
|                         |                |                      | optimizer == SGD       |
+-------------------------+----------------+----------------------+------------------------+
| Number dense nodes      | integer        | [5,40]               | can only take integer  |
|                         |                |                      | values from 5 to 40    |
+-------------------------+----------------+----------------------+------------------------+
| encoder activation      | categorical    | {ReLu, tanh, sigmoid}| discrete choice        |
+-------------------------+----------------+----------------------+------------------------+
| decoder activation      | categorical    | {ReLu, tanh, sigmoid}| discrete choice        |
+-------------------------+----------------+----------------------+------------------------+
| bottleneck activation   | categorical    | {ReLu, tanh, sigmoid}| discrete choice        |
+-------------------------+----------------+----------------------+------------------------+

Please refer to the compute method below to see how those are defined using the
ConfigSpace package.

"""
from numpy.random import seed

seed(123)

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import (
    Dense,
    Lambda,
    Flatten,
    Reshape,
    Input,
    Conv2D,
)


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

from global_utils.get_data_multi_note import read_and_preprocess_data
from global_utils.layers.custom_conv2d_transpose import CustomConv2DTranspose
from global_utils.layers.max_pooling_with_argmax import MaxPoolWithArgMax
from global_utils.layers.unpooling_with_argmax import UnMaxPoolWithArgmax
from global_utils.layers.sampling import sample_from_latent_space

import logging

logging.basicConfig(level=logging.DEBUG)


class KerasWorker(Worker):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.sequence_length = 120
        self.batch_size = 1

        x_train, x_test = read_and_preprocess_data(
            should_smooth=False,
            smoothing_window=100,
            sequence_length=self.sequence_length,
            cut_off_min=5,
            cut_off_max=45,
            should_scale=True,
            batch_size=self.batch_size,
            motes_train=[7],
            motes_test=[7],
        )

        self.x_train = x_train[:310, :, :]
        self.x_test = x_test[310:, :, :]

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        # Encoder
        inputs = Input(shape=(self.sequence_length, 1, 1), batch_size=self.batch_size)

        channels, kernel = 24, 12
        out_conv1 = Conv2D(
            channels,
            (kernel, 1),
            strides=kernel,
            activation=config["encoder_activation"],
            padding="same",
        )(inputs)
        out_reshape1 = Reshape((-1, 1, 1))(out_conv1)
        out_pool1, mask1 = MaxPoolWithArgMax()(out_reshape1)

        channels, kernel = 12, 9
        out_conv2 = Conv2D(
            channels,
            (kernel, 1),
            strides=kernel,
            activation=config["encoder_activation"],
            padding="same",
        )(out_pool1)
        out_reshape2 = Reshape((-1, 1, 1))(out_conv2)
        out_pool2, mask2 = MaxPoolWithArgMax()(out_reshape2)

        out_max_pool_1, mask3 = MaxPoolWithArgMax()(out_pool2)

        channels, kernel = 12, 5
        out_conv3 = Conv2D(
            channels,
            (kernel, 1),
            strides=kernel,
            activation=config["encoder_activation"],
            padding="same",
        )(out_max_pool_1)
        out_reshape3 = Reshape((-1, 1, 1))(out_conv3)
        out_pool3, mask4 = MaxPoolWithArgMax()(out_reshape3)

        channels, kernel = 12, 3
        out_conv4 = Conv2D(
            channels,
            (kernel, 1),
            strides=kernel,
            activation=config["encoder_activation"],
            padding="same",
        )(out_pool3)
        out_reshape4 = Reshape((-1, 1, 1))(out_conv4)
        out_pool4, mask5 = MaxPoolWithArgMax()(out_reshape4)

        out_max_pool_2, mask6 = MaxPoolWithArgMax()(out_pool4)

        out_flatten = Flatten()(out_max_pool_2)
        out_dense1 = Dense(
            config["dense_nodes"], activation=config["bottleneck_activation"]
        )(out_flatten)
        z_mean = Dense(5, activation=config["bottleneck_activation"], name="z_mean")(
            out_dense1
        )
        z_log_var = Dense(
            5, activation=config["bottleneck_activation"], name="z_log_var"
        )(out_dense1)
        z = Lambda(sample_from_latent_space, output_shape=(5,), name="z")(
            [z_mean, z_log_var]
        )

        # Decoder
        de_out_dense2 = Dense(
            config["dense_nodes"], activation=config["bottleneck_activation"]
        )(z)
        de_out_dense3 = Dense(54, activation=config["bottleneck_activation"])(
            de_out_dense2
        )
        de_inverse_flatten = Reshape((-1, 1, 1))(de_out_dense3)

        de_out_pool5 = UnMaxPoolWithArgmax(stride=2)(de_inverse_flatten, mask6)

        channels, kernel = 12, 3
        de_out_pool6 = UnMaxPoolWithArgmax(stride=2)(de_out_pool5, mask5)
        de_out_reshape5 = Reshape(out_conv4.shape[1:])(de_out_pool6)
        de_out_transcov1 = CustomConv2DTranspose(
            channels, kernel, out_pool3.shape, activation=config["decoder_activation"]
        )(de_out_reshape5)

        channels, kernel = 12, 5
        de_out_pool7 = UnMaxPoolWithArgmax(stride=2)(de_out_transcov1, mask4)
        de_out_reshape6 = Reshape(out_conv3.shape[1:])(de_out_pool7)
        de_out_transcov2 = CustomConv2DTranspose(
            channels,
            kernel,
            out_max_pool_1.shape,
            activation=config["decoder_activation"],
        )(de_out_reshape6)

        de_out_pool8 = UnMaxPoolWithArgmax(stride=2)(de_out_transcov2, mask3)

        channels, kernel = 12, 9
        de_out_pool9 = UnMaxPoolWithArgmax(stride=2)(de_out_pool8, mask2)
        de_out_reshape7 = Reshape(out_conv2.shape[1:])(de_out_pool9)
        de_out_transcov3 = CustomConv2DTranspose(
            channels, kernel, out_pool1.shape, activation=config["decoder_activation"]
        )(de_out_reshape7)

        channels, kernel = 24, 12
        de_out_pool10 = UnMaxPoolWithArgmax(stride=2)(de_out_transcov3, mask1)
        de_out_reshape8 = Reshape(out_conv1.shape[1:])(de_out_pool10)
        de_out_transcov4 = CustomConv2DTranspose(
            channels, kernel, inputs.shape, activation=config["decoder_activation"]
        )(de_out_reshape8)
        outputs = de_out_transcov4

        model = Model(inputs, outputs)

        if config["optimizer"] == "Adam":
            optimizer = keras.optimizers.Adam(lr=config["lr"])
        else:
            optimizer = keras.optimizers.SGD(
                lr=config["lr"], momentum=config["sgd_momentum"]
            )

        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=optimizer,
            metrics=[tf.keras.metrics.MeanSquaredError()],
        )

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        tensorboard_log = tf.keras.callbacks.TensorBoard(
            "save_results/CBN_VAE/HpBandSter/tensorboard_logs"
        )

        model.fit(
            self.x_train,
            self.x_train,
            batch_size=self.batch_size,
            epochs=int(budget),
            verbose=1,
            validation_data=(self.x_test, self.x_test),
            callbacks=[stop_early, tensorboard_log],
        )

        # train_score = model.predict(self.x_train, self.x_train, batch_size=self.batch_size)
        # val_score = model.predict(self.x_test, self.x_test, batch_size=self.batch_size)
        train_score = model.evaluate(
            self.x_train, self.x_train, batch_size=self.batch_size, verbose=0
        )

        val_score = model.evaluate(
            self.x_test, self.x_test, batch_size=self.batch_size, verbose=0
        )

        # import IPython; IPython.embed()
        return {
            "loss": val_score[1],  # remember: HpBandSter always minimizes!
            "info": {
                "train accuracy": train_score[1],
                "validation accuracy": val_score[1],
                "number of parameters": model.count_params(),
            },
        }

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter(
            "lr", lower=1e-6, upper=1e-1, default_value="1e-2", log=True
        )

        encoder_activation = CSH.CategoricalHyperparameter(
            "encoder_activation", ["relu", "sigmoid", "tanh"]
        )
        decoder_activation = CSH.CategoricalHyperparameter(
            "decoder_activation", ["relu", "sigmoid", "tanh"]
        )

        bottleneck_activation = CSH.CategoricalHyperparameter(
            "bottleneck_activation", ["relu", "sigmoid", "tanh"]
        )

        # SGD has a different parameter 'momentum'. Coniditonal parameter if SGD is true
        optimizer = CSH.CategoricalHyperparameter("optimizer", ["Adam", "SGD"])

        sgd_momentum = CSH.UniformFloatHyperparameter(
            "sgd_momentum", lower=0.0, upper=0.99, default_value=0.9, log=False
        )

        dense_nodes = CSH.UniformIntegerHyperparameter(
            "dense_nodes", lower=5, upper=40, default_value=27
        )

        cs.add_hyperparameters(
            [
                lr,
                encoder_activation,
                decoder_activation,
                optimizer,
                sgd_momentum,
                dense_nodes,
                bottleneck_activation,
            ]
        )

        return cs


if __name__ == "__main__":
    worker = KerasWorker(run_i1="0")
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(
        config=config,
        budget=5,
        working_directory="/home/paperspace/hyperparameter-optimization/Optimization_HpBandSter/CBN_VAE",
    )
    print(res)
