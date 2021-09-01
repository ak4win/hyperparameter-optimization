""""
hyper-parameter optimization methods Random Search, Bayesian Optimization,
Hyperband and BOHB for the models LSTM-RNN-AE and CBN_VAE
=================
The following variables can be adjusted, just copy and paste the content and set it as argument
+-------------------------+---------------------------------------------------------+------------------------+
| Models                  | possible Optimization method for the Model              |  Range/Choices         |
+=========================+================+========================================+------------------------+
| "CBN_VAE", "RNN"        | "RS" = Random Search, "BO" = Bayesian Optimization,     |  discrete choice       |
|                         | "HB" = Hyperband, "HpBandster" = BOHB method            |                        |
+-------------------------+---------------------------------------------------------+------------------------+
"""
# ======================================================================================================================
# Imports and setup
# ======================================================================================================================
# Set the seed
from numpy.random import seed; seed(1)
import tensorflow as tf; tf.random.set_seed(2)

# Module level imports
from lib.model_configs import model_configs
from lib.get_tuner import get_tuner
from lib.get_best_result_hpbandster import get_best_result
import global_utils.plotter as plotter
from global_utils.plot_sequence import plot_sequence
from global_utils.save_metadata import save_metadata
from global_utils.train_model import retrain_best_model
from global_utils.evaluation import smooth_output
from global_utils.get_data_multi_note import read_and_preprocess_data
from optimization_utils.Optimization_HpBandSter.get_configs import get_num_configs

from lib.create_cbn_vae import create_model

x_train, x_test = read_and_preprocess_data(
    sequence_length=120,
    batch_size=32,
    motes_train=[7],
    motes_test=[7],
)

# split train and test data
train_test_cutoff = 320
x_train = x_train[:train_test_cutoff, :, :]
x_test = x_test[train_test_cutoff:, :, :]

model = create_model({'encoder_activation': 'tanh', 'decoder_activation': 'tanh', 'dense_nodes': 33,
                      'bottleneck_activation': 'relu', 'lr': 0.001521356711612709, 'optimizer': 'Adam',
                      'sgd_momentum': 0.5091287212784572})
x = model(x_train)
y =1 
exit()
# Third party libraries
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
plotter = plotter.Plotter("model", plt)

# ======================================================================================================================
# Initialize wanted model and optimization method - alter the code as desired
# ======================================================================================================================
# choose your model you want to optimize
# "CBN_VAE" or "RNN"
current_model = "CBN_VAE"
model_config = model_configs[current_model]

# choose an optimization method "RS" = Random Search, "BO" == Bayesian Optimization,
# "HB" = Hyperband, "HpBandSter" == Bayesian Optimization Hyperband (BOHB)
current_optimization_method = "HpBandSter"


x_train, x_test = read_and_preprocess_data(
    sequence_length=model_config["sequence_length"],
    batch_size=model_config["batch_size"],
    motes_train=[7],
    motes_test=[7],
)

# split train and test data
train_test_cutoff = model_config["train_test_cutoff"]
x_train = x_train[:train_test_cutoff, :, :]
x_test = x_test[train_test_cutoff:, :, :]
# ======================================================================================================================
# Run the chosen optimization method
# ======================================================================================================================
# check for the optimization method
if current_optimization_method == "HpBandSter":
    run_experiments = model_config["hpbandster_run_experiments_function"]

    t1 = datetime.now()
    run_experiments(x_train, x_test, overwrite=False)
    t2 = datetime.now()
    num_configs = get_num_configs(current_model, current_optimization_method)

    best_config, best_model = get_best_result(current_model)

else:
    create_model = model_config["create_model_fn"]

    tuner = get_tuner(current_optimization_method, create_model, current_model)
    summary_search_space = tuner.search_space_summary()

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    t1 = datetime.now()
    tuner.search(
        x_train,
        x_train,
        epochs=model_config["optimization_epochs"],
        validation_data=(x_test, x_test),
        callbacks=[stop_early] if model_config["should_early_stop"] else [],
        batch_size=model_config["batch_size"],
    )
    t2 = datetime.now()

    best_config = tuner.get_best_hyperparameters()[0].values
    best_model = tuner.get_best_models()[0]
    num_configs = len(tuner.get_best_models(num_models=10000))

# ======================================================================================================================
# Retrain the best model and contextualize the results
# ======================================================================================================================
# Retrain the model on the best
t3 = datetime.now()
history, train_preds, test_preds, model_after_training = retrain_best_model(
    best_model, x_train, x_test, model_config
)
t4 = datetime.now()

plot_sequence(
    (np.array(history["loss"]), "Training loss"),
    (np.array(history["val_loss"]), "Validation loss"),
)
save_metadata(current_model, current_optimization_method, t2 - t1, t4 - t3, num_configs)

plotter("learning_curve")


# Save the model for later use
file_path = f"/home/paperspace/hyperparameter-optimization/save_models/{current_model}"
tf.keras.models.save_model(model_after_training, file_path, overwrite=True)

# Smooth the outputs
# This step was implemented to make up for the noisy reconstructions that canhappen in some cases.
# It however only slightly improves the prms-difference.
reconstruction, prms_diff = smooth_output(
    x_test, test_preds, smoothing_window=model_config["rolling_avg_smoothing_window"]
)

# Get a summary of the results
print(
    f'Algorithm "{current_optimization_method}" found this config for model "{current_model}": {best_config}'
)
print(
    f"The percentual-RMS-difference for the configuration after the re-training is {prms_diff}"
)


print(
    f"Optimizing took {t2-t1} | Tried {num_configs} configurations | Retraining took {t4-t3}"
)

# Visualize reconstructions for more intuitive judgement of the results
plot_sequence(
    (x_test.reshape(-1), "original"),
    (reconstruction.reshape(-1), "reconstruction"),
    title=prms_diff,
)
plotter("reconstruction_1k")

plot_sequence(
    (x_test.reshape(-1)[:4000], "original"),
    (reconstruction.reshape(-1)[:4000], "reconstruction"),
    title=prms_diff,
)
plotter("reconstruction_4k")
# ======================================================================================================================
# EOF
# ======================================================================================================================
