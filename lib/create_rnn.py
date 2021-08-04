from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.models import Model
from keras.optimizers import Adam, SGD
from keras.losses import MeanSquaredError


def create_model(config, batch_size=1, sequence_length=20, n_dims=1):
    inputs = Input(
        shape=(sequence_length, n_dims), batch_size=batch_size
    )
    x = inputs
    x = LSTM(
        units=n_dims,
        return_sequences=False,
        stateful=True,
        activation=config["activation_encoder"],
        recurrent_activation=config["recurrent_activation_encoder"],
        # kernel_regularizer=kernel_regularizer,
        # bias_regularizer=bias_regularizer,
        # activity_regularizer=activity_regularizer,
        # dropout=config["dropout"],
        # recurrent_dropout=config["recurrent_dropout"],
    )(x)

    x = RepeatVector(sequence_length)(x)
    x = LSTM(
        units=n_dims,
        stateful=True,
        return_sequences=True,
        # bias_initializer=global_mean_bias,
        # unit_forget_bias=False,
        activation=config["activation_decoder"],
        recurrent_activation=config["recurrent_activation_decoder"],
        # kernel_regularizer=kernel_regularizer,
        # bias_regularizer=bias_regularizer,
        # activity_regularizer=activity_regularizer,
        # dropout=config["dropout"],
        # recurrent_dropout=config["recurrent_dropout"],
    )(x)

    outputs = x

    model = Model(inputs, outputs)

    if config["optimizer"] == "Adam":
        optimizer = Adam(lr=config["lr"])
    else:
        optimizer = SGD(
            lr=config["lr"], momentum=config["sgd_momentum"]
        )

    model.compile(
        loss=MeanSquaredError(),
        optimizer=optimizer,
        metrics=[],
    )

    return model
