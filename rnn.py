from numpy.random import seed

seed(1)

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_model_rnn(hp):

    sequence_length = 20
    n_dims = 1
    batch_size = 1
    activation = hp.Choice("activation", ["tanh"])
    recurrent_activation = hp.Choice("reccurent_activation", ["sigmoid"])

    # kernel_regularizer=l2(0.0),
    # bias_regularizer=l2(0.0),
    # activity_regularizer=l2(0.0),
    dropout = 0.0
    recurrent_dropout = 0.0

    # [batch, timesteps, feature] is shape of inputs
    inputs = Input(shape=(sequence_length, n_dims), batch_size=batch_size)
    x = inputs
    x = LSTM(
        n_dims,
        return_sequences=False,
        stateful=True,
        activation=activation,
        recurrent_activation=recurrent_activation,
        # kernel_regularizer=kernel_regularizer,
        # bias_regularizer=bias_regularizer,
        # activity_regularizer=activity_regularizer,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
    )(x)

    x = RepeatVector(sequence_length)(x)
    x = LSTM(
        n_dims,
        stateful=True,
        return_sequences=True,
        # bias_initializer=global_mean_bias,
        unit_forget_bias=False,
        activation=activation,
        recurrent_activation=recurrent_activation,
        # kernel_regularizer=kernel_regularizer,
        # bias_regularizer=bias_regularizer,
        # activity_regularizer=activity_regularizer,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
    )(x)

    outputs = x

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
        loss=hp.Choice("loss_function", hp.Choice("loss_function", ["mse"])),
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )

    return model
