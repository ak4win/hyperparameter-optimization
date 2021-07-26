"""
Worker for Example 5 - Keras
============================
In this example implements a small CNN in Keras to train it on MNIST.
The configuration space shows the most common types of hyperparameters and
even contains conditional dependencies.
We'll optimise the following hyperparameters:
+-------------------------+----------------+-----------------+------------------------+
| Parameter Name          | Parameter type |  Range/Choices  | Comment                |
+=========================+================+=================+========================+
| Learning rate           |  float         | [1e-6, 1e-2]    | varied logarithmically |
+-------------------------+----------------+-----------------+------------------------+
| Optimizer               | categorical    | {Adam, SGD }    | discrete choice        |
+-------------------------+----------------+-----------------+------------------------+
| SGD momentum            |  float         | [0, 0.99]       | only active if         |
|                         |                |                 | optimizer == SGD       |
+-------------------------+----------------+-----------------+------------------------+
| Number of conv layers   | integer        | [1,3]           | can only take integer  |
|                         |                |                 | values 1, 2, or 3      |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | logarithmically varied |
| the first conf layer    |                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the second conf layer   |                |                 | of layers >= 2         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the third conf layer    |                |                 | of layers == 3         |
+-------------------------+----------------+-----------------+------------------------+
| Dropout rate            |  float         | [0, 0.9]        | standard continuous    |
|                         |                |                 | parameter              |
+-------------------------+----------------+-----------------+------------------------+
| Number of hidden units  | integer        | [8,256]         | logarithmically varied |
| in fully connected layer|                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+
Please refer to the compute method below to see how those are defined using the
ConfigSpace package.

The network does not achieve stellar performance when a random configuration is samples,
but a few iterations should yield an accuracy of >90%. To speed up training, only
8192 images are used for training, 1024 for validation.
The purpose is not to achieve state of the art on MNIST, but to show how to use
Keras inside HpBandSter, and to demonstrate a more complicated search space.
"""

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Input, RepeatVector
from keras.layers import LSTM
from keras import backend as K


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

from global_utils.get_data_multi_note import read_and_preprocess_data


import logging

logging.basicConfig(level=logging.DEBUG)


class KerasWorker(Worker):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.batch_size = 1
        self.sequence_length = 20
        self.n_dims = 1

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

        self.x_train = x_train[:1900, :, :]
        self.x_test = x_test[1900:, :, :]

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        inputs = Input(
            shape=(self.sequence_length, self.n_dims), batch_size=self.batch_size
        )
        x = inputs
        x = LSTM(
            units=self.n_dims,
            return_sequences=False,
            stateful=True,
            activation=config["activation"],
            recurrent_activation=config["recurrent_activation"],
            # kernel_regularizer=kernel_regularizer,
            # bias_regularizer=bias_regularizer,
            # activity_regularizer=activity_regularizer,
            # dropout=dropout,
            # recurrent_dropout=recurrent_dropout,
        )(x)

        x = RepeatVector(self.sequence_length)(x)
        x = LSTM(
            units=self.n_dims,
            stateful=True,
            return_sequences=True,
            # bias_initializer=global_mean_bias,
            unit_forget_bias=False,
            activation=config["activation"],
            recurrent_activation=config["recurrent_activation"],
            # kernel_regularizer=kernel_regularizer,
            # bias_regularizer=bias_regularizer,
            # activity_regularizer=activity_regularizer,
            # dropout=dropout,
            # recurrent_dropout=recurrent_dropout,
        )(x)

        outputs = x

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

        model.fit(
            self.x_train,
            self.x_train,
            batch_size=self.batch_size,
            epochs=int(budget),
            verbose=1,
            validation_data=(self.x_test, self.x_test),
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
            "loss": val_score,  # remember: HpBandSter always minimizes!
            "info": {
                "train accuracy": train_score,
                "validation accuracy": val_score,
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

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        activation = CSH.CategoricalHyperparameter("activation", ["relu", "sigmoid"])
        recurrent_activation = CSH.CategoricalHyperparameter(
            "recurrent_activation", ["sigmoid"]
        )

        optimizer = CSH.CategoricalHyperparameter("optimizer", ["Adam", "SGD"])

        sgd_momentum = CSH.UniformFloatHyperparameter(
            "sgd_momentum", lower=0.0, upper=0.99, default_value=0.9, log=False
        )

        cs.add_hyperparameters(
            [lr, activation, recurrent_activation, optimizer, sgd_momentum]
        )

        # num_conv_layers = CSH.UniformIntegerHyperparameter(
        #     "num_conv_layers", lower=1, upper=3, default_value=2
        # )

        # num_filters_1 = CSH.UniformIntegerHyperparameter(
        #     "num_filters_1", lower=4, upper=64, default_value=16, log=True
        # )
        # num_filters_2 = CSH.UniformIntegerHyperparameter(
        #     "num_filters_2", lower=4, upper=64, default_value=16, log=True
        # )
        # num_filters_3 = CSH.UniformIntegerHyperparameter(
        #     "num_filters_3", lower=4, upper=64, default_value=16, log=True
        # )

        # cs.add_hyperparameters(
        #     [num_conv_layers, num_filters_1, num_filters_2, num_filters_3]
        # )

        # dropout_rate = CSH.UniformFloatHyperparameter(
        #     "dropout_rate", lower=0.0, upper=0.9, default_value=0.5, log=False
        # )
        # num_fc_units = CSH.UniformIntegerHyperparameter(
        #     "num_fc_units", lower=8, upper=256, default_value=32, log=True
        # )

        # cs.add_hyperparameters([dropout_rate, num_fc_units])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, "SGD")
        cs.add_condition(cond)

        # You can also use inequality conditions:
        # cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
        # cs.add_condition(cond)

        # cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
        # cs.add_condition(cond)

        return cs


if __name__ == "__main__":
    worker = KerasWorker(run_id="0")
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(
        config=config,
        budget=5,
        working_directory="/home/paperspace/hyperparameter-optimization/HpBandSter",
    )
    print(res)
