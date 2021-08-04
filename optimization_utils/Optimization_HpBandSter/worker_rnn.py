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

from lib.create_rnn import create_model
from global_utils.plot_sequence import plot_sequence
from global_utils.plotter import Plotter
import tensorflow as tf

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

from global_utils.get_data_multi_note import read_and_preprocess_data

import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.WARNING)
plotter = Plotter('debug_rnn', plt)


class KerasWorker(Worker):
    def __init__(self, x_train, x_test, **kwargs):

        super().__init__(**kwargs)

        self.batch_size = 1
        self.sequence_length = 20
        self.n_dims = 1

        self.x_train = x_train
        self.x_test = x_test

    def compute(self, config, budget, working_directory, *args, **kwargs):
        model = create_model(config)

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

        plot_sequence((model.predict(self.x_train, batch_size=self.batch_size),
                      'reconstruction'), (self.x_train, 'originial'))
        plotter('train')
        plot_sequence((model.predict(self.x_test, batch_size=self.batch_size),
                      'reconstruction'), (self.x_test, 'originial'))
        plotter('test')

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
            "activation_encoder", ["relu", "sigmoid", "tanh"]
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

        # dropout = CSH.UniformFloatHyperparameter(
        #     "dropout", lower=0.0, upper=0.3, default_value=0.0
        # )

        # recurrent_dropout = CSH.UniformFloatHyperparameter(
        #     "recurrent_dropout", lower=0.0, upper=0.3, default_value=0.0
        # )

        # cs.add_hyperparameters([dropout, recurrent_dropout])

        # The hyperparameter sgd_momentum will be used, if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, "SGD")
        cs.add_condition(cond)

        return cs


if __name__ == "__main__":
    x_train, x_test = read_and_preprocess_data(
        sequence_length=20,
        batch_size=1,
        motes_train=[7],
        motes_test=[7],
    )
    # x_train = x_train[:100, :, :]
    # x_test = x_test[200:225, :, :]
    x_train = x_train[:1900, :, :]
    x_test = x_test[1900:, :, :]

    worker = KerasWorker(x_train, x_test, run_id="0")
    # cs = worker.get_configspace()
    # config = cs.sample_configuration().get_dictionary()
    config = {'optimizer': 'Adam', 'lr': 0.001,
              'recurrent_activation_encoder': 'sigmoid', 'activation_encoder': 'tanh',
              'recurrent_activation_decoder': 'sigmoid', 'activation_decoder': 'tanh',
              'dropout': 0.0, 'recurrent_dropout': 0.0
              }

    print(config)
    res = worker.compute(
        config=config,
        budget=1,
        working_directory="/home/paperspace/hyperparameter-optimization/Optimization_HpBandSter/RNN",
    )
    print(res)
