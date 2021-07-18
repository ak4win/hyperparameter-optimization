# Set the seed
from numpy.random import seed

seed(1)

# import numpy and tf
import numpy as np
import tensorflow as tf

tf.random.set_seed(2)

# machine learning libaries
import keras.backend as K

# hyper-parameter optimization
from keras_tuner import Hyperband, RandomSearch, HyperParameters

# own modules layers
from global_utils.get_data_multi_note_without_smoothing import get_data
from global_utils.evaluation import per_rms_diff, smooth_output
from model import create_model

x_train, x_test = get_data()
sample_size = 120
batch_size = 32
smoothing_true = True

model = create_model()

tuner = RandomSearch(
    model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory="test_random_search"
    # project_name="test_random_seach_v1",
)

summary_search_space = tuner.search_space_summary()

optimization = tuner.search(
    x_train, x_train, epochs=20, validation_data=(x_test, x_test)
)

best_model = tuner.get_best_models()[0]

val_predictions = best_model.predict(x_test)

if smoothing_true is True:
    evaluation = smooth_output(val_predictions)
else:
    evaluation = per_rms_diff(x_test, val_predictions)
