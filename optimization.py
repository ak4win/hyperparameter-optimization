# Set the seed
from numpy.random import seed

seed(1)

# import numpy and tf
import tensorflow as tf

tf.random.set_seed(2)

# Plotting libaries
import matplotlib.pyplot as plt


# hyper-parameter optimization
from keras_tuner import RandomSearch, BayesianOptimization, Hyperband

# own modules layers
from global_utils.get_data_multi_note_without_smoothing import get_data
from global_utils.evaluation import per_rms_diff, smooth_output
from model import create_model
import global_utils.plotter as plotter

plot = plotter.Plotter("VAE", plt)

x_train, x_test = get_data()
sample_size = 120
batch_size = 32
smoothing_true = True

the_choice = "HP"

optimization_method = {
    "RS": RandomSearch(
        create_model,  # model instance, whose hyper-parameters are optimized
        objective="val_loss",  # the direction of the optimization
        max_trials=2,  # the max. amount of model configurations that are tested
        executions_per_trial=3,  # how many rounds the model with that configuration is trained to reduce variance
        # seed=1,  # set the seed
        overwrite=True,  # boolean whether to overwrite the project
        directory="test_random_search"  # the relative path to the working directory
        # project_name="test_random_seach_v1",
    ),
    "BO": BayesianOptimization(
        create_model,  # model instance, whose hyper-parameters are optimized
        objective="val_loss",  # the direction of the optimization
        max_trials=10,  # the max. amount of model configurations that are tested
        num_initial_points=3,  # number of randomly generated samples
        alpha=1e-4,  # the expected amount of noise in the observed performances
        beta=2.6,  # factor to balance exploration and explotation
        executions_per_trial=1,  # how many rounds the model with that configuration is trained to reduce variance
        # seed=1,  # set the seed
        overwrite=True,  # boolean whether to overwrite the project
        directory="test_bayesian_optimization"  # the relative path to the working directory
        # project_name="test_random_seach_v1",
    ),
    "HP": Hyperband(
        create_model,  # model instance, whose hyper-parameters are optimized
        objective="val_loss",  # the direction of the optimization
        max_epochs=25,  # the max. amount of model configurations that are tested
        factor=3,  # reduction factor for the number of epochs and number of models for each bracket
        hyperband_iterations=2,  # number of times to iterate over the full Hyperband algorithm
        executions_per_trial=1,  # how many rounds the model with that configuration is trained to reduce variance
        # seed=1,  # set the seed
        overwrite=True,  # boolean whether to overwrite the project
        directory="test_hyperband_optimization",
    ),
}[the_choice]

tuner = optimization_method

# if optimization_method is RandomSearch:
#     tuner = optimization_method(
#         create_model,
#         objective="val_loss",
#         max_trials=10,
#         executions_per_trial=3,
#         overwrite=True,
#         directory="test_random_search"
# project_name="test_random_seach_v1",
# )
# elif optimization_method is BayesianOptimization:

# elif optimization is Hyperband:


summary_search_space = tuner.search_space_summary()

stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

optimization = tuner.search(
    x_train,
    x_train,
    epochs=25,
    validation_data=(x_test, x_test),
    callbacks=[stop_early],
)

best_model = tuner.get_best_models()[0]

val_predictions = best_model.predict(x_test)

# if smoothing_true is True:
diff, evaluation = smooth_output(x_test, val_predictions, smoothing_window=3)
print(evaluation)
# else:
evaluation = per_rms_diff(x_test.reshape(-1), val_predictions)
# print(evaluation)

plt.plot(x_test.reshape(-1)[3000:4000], label="test-data")
plt.plot(diff[3000:4000], label="reconstruction")
plt.legend()
plot("reconstruction")