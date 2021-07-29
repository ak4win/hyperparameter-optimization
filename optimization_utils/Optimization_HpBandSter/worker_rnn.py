"""
LSTM-RNN-Auto-Encoder
=================
Optimization of the LSTM-RNN Model which is implemented
in our research project.

The configuration space shows the types of hyperparameters and
even contains conditional dependencies.

We'll optimise the following hyperparameters:
+-------------------------+----------------+----------------------+------------------------+
| Parameter Name          | Parameter type |  Range/Choices       | Comment                |
+=========================+================+======================+========================+
| Learning rate           | float          | [1e-6, 1e-2]         | varied logarithmically |
+-------------------------+----------------+----------------------+------------------------+
| Optimizer               | categorical    | {Adam, SGD}          | discrete choice        |
+-------------------------+----------------+----------------------+------------------------+
| SGD momentum            | float          | [0, 0.99]            | only active if         |
|                         |                |                      | optimizer == SGD       |
+-------------------------+----------------+----------------------+------------------------+
| dropout rate encoder    | float          | [0, 0.3]             | continuous value       |
| and decoder             |                |                      |                        |
+-------------------------+----------------+----------------------+------------------------+
| recurrent dropout rate  | float          | [0, 0.3]             | continuous value       |
| encoder and decoder     |                |                      |                        |
+-------------------------+----------------+----------------------+------------------------+
| encoder activation      | categorical    | {ReLu, tanh, sigmoid}| discrete choice        |
+-------------------------+----------------+----------------------+------------------------+
| decoder activation      | categorical    | {ReLu, tanh, sigmoid}| discrete choice        |
+-------------------------+----------------+----------------------+------------------------+
| recurrent encoder       | categorical    | {ReLu, tanh, sigmoid}| discrete choice        |
| activation              |                |                      |                        |
+-------------------------+----------------+----------------------+------------------------+
| recurrent decoder       | categorical    | {ReLu, tanh, sigmoid}| discrete choice        |
| activation              |                |                      |                        |
+-------------------------+----------------+----------------------+------------------------+

Please refer to the compute method below to see how those are defined using the
ConfigSpace package.
"""

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Input, RepeatVector
from keras.layers import LSTM


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
            activation=config["encoder_activation"],
            recurrent_activation=config["recurrent_activation_encoder"],
            # kernel_regularizer=kernel_regularizer,
            # bias_regularizer=bias_regularizer,
            # activity_regularizer=activity_regularizer,
            dropout=config["dropout"],
            recurrent_dropout=config["recurrent_dropout"],
        )(x)

        x = RepeatVector(self.sequence_length)(x)
        x = LSTM(
            units=self.n_dims,
            stateful=True,
            return_sequences=True,
            # bias_initializer=global_mean_bias,
            unit_forget_bias=False,
            activation=config["activation_decoder"],
            recurrent_activation=config["recurrent_activation_decoder"],
            # kernel_regularizer=kernel_regularizer,
            # bias_regularizer=bias_regularizer,
            # activity_regularizer=activity_regularizer,
            dropout=config["dropout"],
            recurrent_dropout=config["recurrent_dropout"],
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

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        tensorboard_log = tf.keras.callbacks.TensorBoard(
            "save_results/RNN/HpBandSter/tensorboard_logs"
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
        activation_decoder = CSH.CategoricalHyperparameter(
            "activation_decoder", ["relu", "sigmoid", "tanh"]
        )

        recurrent_activation_encoder = CSH.CategoricalHyperparameter(
            "recurrent_activation_encoder", ["relu", "sigmoid", "tanh"]
        )
        recurrent_activation_decoder = CSH.CategoricalHyperparameter(
            "recurrent_activation_decoder", ["relu", "sigmoid", "tanh"]
        )

        optimizer = CSH.CategoricalHyperparameter("optimizer", ["Adam", "SGD"])

        # SGD has a different parameter 'momentum'.
        sgd_momentum = CSH.UniformFloatHyperparameter(
            "sgd_momentum", lower=0.0, upper=0.99, default_value=0.9, log=False
        )

        cs.add_hyperparameters(
            [
                lr,
                encoder_activation,
                activation_decoder,
                recurrent_activation_encoder,
                recurrent_activation_decoder,
                optimizer,
                sgd_momentum,
            ]
        )

        dropout = CSH.UniformFloatHyperparameter(
            "dropout", lower=0.0, upper=0.3, default_value=0.0
        )

        recurrent_dropout = CSH.UniformFloatHyperparameter(
            "recurrent_dropout", lower=0.0, upper=0.3, default_value=0.0
        )

        cs.add_hyperparameters([dropout, recurrent_dropout])

        # The hyperparameter sgd_momentum will be used, if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, "SGD")
        cs.add_condition(cond)

        return cs


if __name__ == "__main__":
    worker = KerasWorker(run_id="0")
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(
        config=config,
        budget=5,
        working_directory="/home/paperspace/hyperparameter-optimization/Optimization_HpBandSter/RNN",
    )
    print(res)
