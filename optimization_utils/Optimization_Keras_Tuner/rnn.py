from lib.create_rnn import create_model
from numpy.random import seed

seed(1)
import tensorflow as tf

tf.random.set_seed(2)


def create_model_rnn(hp):
    activation_encoder = hp.Choice("activation_encoder", ["tanh", "relu", "sigmoid"])
    recurrent_activation_encoder = hp.Choice(
        "recurrent_activation_encoder", ["tanh", "relu", "sigmoid"]
    )
    activation_decoder = hp.Choice("activation_decoder", ["tanh", "relu", "sigmoid"])
    recurrent_activation_decoder = hp.Choice(
        "recurrent_activation_decoder", ["tanh", "relu", "sigmoid"]
    )

    dropout = hp.Float("dropout", min_value=0.0, max_value=0.3)
    recurrent_dropout = hp.Float("recurrent_dropout", min_value=0.0, max_value=0.3)
    learning_rate = hp.Float("learning_rate", min_value=1e-6, max_value=1e-2)
    optimizer = hp.Choice("optimizer", ["Adam", "SGD"])
    sgd_momentum = hp.Float("sgd_momentum", min_value=0.01, max_value=0.99)

    config = {
        "recurrent_activation_encoder": recurrent_activation_encoder,
        "activation_encoder": activation_encoder,
        "activation_decoder": activation_decoder,
        "recurrent_activation_decoder": recurrent_activation_decoder,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "lr": learning_rate,
        "optimizer": optimizer,
        "sgd_momentum": sgd_momentum,
    }

    return create_model(config)
