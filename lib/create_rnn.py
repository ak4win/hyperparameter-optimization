from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from lib.mse_without_nans import mean_squared_error_without_nans


def create_model(config: dict, batch_size: int = 1, sequence_length: int = 20, n_dims: int = 1) -> Model:
    inputs = Input(shape=(sequence_length, n_dims), batch_size=batch_size)
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
        dropout=config["dropout"],
        recurrent_dropout=config["recurrent_dropout"],
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
        dropout=config["dropout"],
        recurrent_dropout=config["recurrent_dropout"],
    )(x)

    outputs = x

    model = Model(inputs, outputs)

    if config["optimizer"] == "Adam":
        optimizer = Adam(lr=config["lr"])
    else:
        optimizer = SGD(lr=config["lr"], momentum=config["sgd_momentum"])

    model.compile(
        loss=mean_squared_error_without_nans,
        optimizer=optimizer,
        metrics=[],
    )

    return model
