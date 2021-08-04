from lib.create_rnn import create_model
from numpy.random import seed; seed(1)
import tensorflow as tf; tf.random.set_seed(2)


def create_model_rnn(hp):
    activation_encoder = hp.Choice("activation_encoder", ["tanh", "relu", "sigmoid"])
    activation_decoder = hp.Choice("activation_decoder", ["tanh", "relu", "sigmoid"])
    recurrent_activation_encoder = hp.Choice("recurrent_activation_encoder", ["sigmoid", "tanh"])
    recurrent_activation_decoder = hp.Choice("recurrent_activation_decoder", ["sigmoid", "tanh"])
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    config = {'optimizer': 'Adam', 'lr': learning_rate,
              'recurrent_activation_encoder': recurrent_activation_encoder, 'activation_encoder': activation_encoder,
              'recurrent_activation_decoder': recurrent_activation_decoder, 'activation_decoder': activation_decoder,
              'dropout': 0.0, 'recurrent_dropout': 0.0
              }
    return create_model(config)
