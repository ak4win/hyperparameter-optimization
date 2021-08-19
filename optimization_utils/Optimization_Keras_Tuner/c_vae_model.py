from lib.create_cbn_vae import create_model
from numpy.random import seed

seed(1)
import tensorflow as tf

tf.random.set_seed(2)


def create_model_c_vae(hp):

    encoder_activation = hp.Choice("encoder_activation", ["relu", "tanh", "sigmoid"])
    decoder_activation = hp.Choice("decoder_activation", ["relu", "tanh", "sigmoid"])
    dense_nodes = hp.Int("dense_nodes", min_value=5, max_value=40)
    bottleneck_activation = hp.Choice(
        "bottleneck_activation", ["relu", "tanh", "sigmoid"]
    )

    learning_rate = hp.Float("learning_rate", min_value=1e-6, max_value=1e-2)
    optimizer = hp.Choice("optimizer", ["Adam", "SGD"])
    sgd_momentum = hp.Float("sgd_momentum", min_value=0.01, max_value=0.99)

    config = {
        "encoder_activation": encoder_activation,
        "decoder_activation": decoder_activation,
        "dense_nodes": dense_nodes,
        "bottleneck_activation": bottleneck_activation,
        "lr": learning_rate,
        "optimizer": optimizer,
        "sgd_momentum": sgd_momentum,
    }

    return create_model(config)
