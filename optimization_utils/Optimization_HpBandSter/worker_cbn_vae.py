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

seed(1)
import tensorflow as tf

tf.random.set_seed(2)


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

from global_utils.get_data_multi_note import read_and_preprocess_data
from global_utils.plot_sequence import plot_sequence
from lib.create_cbn_vae import create_model

import logging

from global_utils.plotter import Plotter
import matplotlib.pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.WARNING)
plotter = Plotter("debug_cbn", plt)


class KerasWorker(Worker):
    def __init__(self, x_train, x_test, **kwargs):

        super().__init__(**kwargs)

        self.sequence_length = 120
        self.batch_size = 1

        self.x_train = x_train
        self.x_test = x_test

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        model = create_model(
            config, batch_size=self.batch_size, sequence_length=self.sequence_length
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
            callbacks=[
                tensorboard_log
            ],  # Don't early stop for now as the learning curve is rather noisy
        )
        plot_sequence(
            (model.predict(self.x_train, batch_size=self.batch_size), "reconstr"),
            (self.x_train, "original"),
        )
        plotter("train")
        plot_sequence(
            (model.predict(self.x_test, batch_size=self.batch_size), "reconstr"),
            (self.x_test, "original"),
        )
        plotter("test")

        train_score = model.evaluate(
            self.x_train, self.x_train, batch_size=self.batch_size, verbose=0
        )

        val_score = model.evaluate(
            self.x_test, self.x_test, batch_size=self.batch_size, verbose=0
        )

        # import IPython; IPython.embed()
        return {
            "loss": val_score,  # remember: HpBandSter always minimizes!
            "info": {
                "train loss": train_score,
                "validation loss": val_score,
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
    x_train, x_test = read_and_preprocess_data(
        sequence_length=120,
        batch_size=32,
        motes_train=[7],
        motes_test=[7],
    )
    x_train = x_train[32:64, :, :]
    x_test = x_test[64:96, :, :]

    worker = KerasWorker(x_train, x_test, run_id="0")
    cs = worker.get_configspace()

    # config = cs.sample_configuration().get_dictionary()
    # print(config)
    # config = {"bottleneck_activation": "relu", "decoder_activation": "tanh", "dense_nodes": 26,
    #           "encoder_activation": "tanh", "lr": 0.007220465601391944, "optimizer": "Adam",
    #           "sgd_momentum": 0.6499231840785764}

    # Changing decoder activation from sigmoid to tanh will improve results drastically

    config = {
        "bottleneck_activation": "relu",
        "decoder_activation": "sigmoid",
        "dense_nodes": 12,
        "encoder_activation": "tanh",
        "lr": 0.0046277046865170245,
        "optimizer": "Adam",
        "sgd_momentum": 0.9212735787477947,
    }

    res = worker.compute(
        config=config,
        budget=25,
        working_directory="/home/paperspace/hyperparameter-optimization/Optimization_HpBandSter/CBN_VAE",
    )
    print(res)
