from numpy.random import seed

seed(1)

# Tensorflow imports
from tensorflow.keras.layers import LSTM, Input, RepeatVector
from tensorflow.keras.models import Model


def create_model(hp):

    sequence_length = hp.Choice("encoder_activation", ["relu", "tanh"])
    n_dims = hp.Choice("encoder_activation", ["relu", "tanh"])
    batch_size = hp.Choice("encoder_activation", ["relu", "tanh"])
    activation = "tanh"
    recurrent_activation = "sigmoid"
    # kernel_regularizer=l2(0.0),
    # bias_regularizer=l2(0.0),
    # activity_regularizer=l2(0.0),
    dropout = 0.0
    recurrent_dropout = 0.0

    # [batch, timesteps, feature] is shape of inputs
    inputs: Input = Input(shape=(sequence_length, n_dims), batch_size=batch_size)
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
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model
