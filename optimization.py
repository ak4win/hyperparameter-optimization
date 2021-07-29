""""
hyper-parameter optimization methods Random Search, Bayesian Optimization,
Hyperband and BOHB for the models LSTM-RNN-AE and CBN-VAE
=================
The following variables can be adjusted, just copy and past the content and set it as argument
+-------------------------+---------------------------------------------------------+------------------------+
| Models                  | possible Optimization method for the Model              |  Range/Choices         |
+=========================+================+========================================+------------------------+
| "CBN-VAE", "RNN"        | "RS" = Random Search, "BO" == Bayesian Optimization,    |  discrete choice       |
|                         | "HB" = Hyperband, "HpBandster" == BOHB method           |                        |
+-------------------------+---------------------------------------------------------+------------------------+
"""

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
from global_utils.get_data_multi_note import read_and_preprocess_data
from global_utils.evaluation import smooth_output
from global_utils.train_model import train_best_model
import global_utils.plotter as plotter

# prepare Plot module for visulization
plot = plotter.Plotter("model", plt)

# choose your model you want to optimize
# "CBN-VAE" or "RNN"
the_model = "CBN-VAE"

# choose an optimization method "RS" = Random Search, "BO" == Bayesian Optimization,
# "HB" = Hyperband, "HpBandster" == BOHB method
the_optimization_method = "HB"
smoothing_true = True
epochs = 32

# check for the optimization method
if the_optimization_method == "HpBandSter":
    # check for the model
    if the_model == "RNN":
        # import corresponding model
        from optimization_utils.Optimization_HpBandSter.worker_rnn import KerasWorker
    else:
        # import corresponding model
        from optimization_utils.Optimization_HpBandSter.worker_cbn_vae import (
            KerasWorker,
        )
    # create worker
    worker = KerasWorker(run_id="0")
    # get configspace for corresponding model
    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    print(config)
    # start worker
    res = worker.compute(
        config=config,
        budget=5,
        working_directory=f"/home/paperspace/hyperparameter-optimization/{the_optimization_method}/{the_model}",
    )
    print(res)

else:
    # check for model
    if the_model == "RNN":
        # import corresponding model
        from optimization_utils.Optimization_Keras_Tuner.rnn import create_model_rnn

        # set variables for Model, Model utils and Optimization
        batch_size = 1
        sequence_length = 20

        x_train, x_test = read_and_preprocess_data(
            should_smooth=False,
            smoothing_window=100,
            sequence_length=sequence_length,
            cut_off_min=5,
            cut_off_max=45,
            should_scale=True,
            batch_size=batch_size,
            motes_train=[7],
            motes_test=[7],
        )

        # split train and test data
        x_train = x_train[:1900, :, :]
        x_test = x_test[1900:, :, :]

        # create model instance
        create_model = create_model_rnn

    else:
        # import corresponding model
        from optimization_utils.Optimization_Keras_Tuner.c_vae_model import (
            create_model_c_vae,
        )

        # set variables for Model, Model utils and Optimization
        sequence_length = 120
        batch_size = 1

        x_train, x_test = read_and_preprocess_data(
            should_smooth=False,
            smoothing_window=100,
            sequence_length=sequence_length,
            cut_off_min=5,
            cut_off_max=45,
            should_scale=True,
            batch_size=batch_size,
            motes_train=[7],
            motes_test=[7],
        )

        x_train = x_train[:310, :, :]
        x_test = x_test[310:, :, :]

        create_model = create_model_c_vae

    file_path = f"/home/paperspace/hyperparameter-optimization/save_models/{the_model}"

    optimization_method = {
        "RS": RandomSearch(
            create_model,  # model instance, whose hyper-parameters are optimized
            objective="val_loss",  # the direction of the optimization
            max_trials=1,  # the max. amount of model configurations that are tested
            executions_per_trial=1,  # how many rounds the model with that configuration is trained to reduce variance
            # seed=1,  # set the seed
            overwrite=True,  # boolean whether to overwrite the project
            directory=f"save_results/{the_model}/random_search",  # the relative path to the working directory
            project_name="results",
        ),
        "BO": BayesianOptimization(
            create_model,  # model instance, whose hyper-parameters are optimized
            objective="val_loss",  # the direction of the optimization
            max_trials=10,  # the max. amount of model configurations that are tested
            num_initial_points=3,  # number of randomly generated samples
            alpha=1e-4,  # the expected amount of noise in the observed performances
            beta=2.6,  # factor to balance exploration and explotation
            executions_per_trial=2,  # how many rounds the model with that configuration is trained to reduce variance
            # seed=1,  # set the seed
            overwrite=True,  # boolean whether to overwrite the project
            directory=f"save_results/{the_model}/bayesian",  # the relative path to the working directory
            project_name="results",
        ),
        "HB": Hyperband(
            create_model,  # model instance, whose hyper-parameters are optimized
            objective="val_loss",  # the direction of the optimization
            max_epochs=15,  # the max. amount of model configurations that are tested
            factor=3,  # reduction factor for the number of epochs and number of models for each bracket
            hyperband_iterations=3,  # number of times to iterate over the full Hyperband algorithm
            executions_per_trial=1,  # how many rounds the model with that configuration is trained to reduce variance
            # seed=1,  # set the seed
            overwrite=True,  # boolean whether to overwrite the project
            directory=f"save_results/{the_model}/hyperband",
            project_name="results",
        ),
    }[the_optimization_method]

    tuner = optimization_method

    summary_search_space = tuner.search_space_summary()

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    optimization = tuner.search(
        x_train,
        x_train,
        epochs=epochs,
        validation_data=(x_test, x_test),
        callbacks=[stop_early],
        batch_size=batch_size,
    )
    best_hp = tuner.get_best_hyperparameters()[0]

    best_model = tuner.get_best_models()[0]

    history, train_preds, test_preds, model_after_training = train_best_model(
        best_model, x_train, x_test, batch_size, epochs=40
    )

    tf.keras.models.save_model(model_after_training, file_path, overwrite=True)

    diff, evaluation = smooth_output(x_test, test_preds, smoothing_window=3)

    print(
        f"The best hyperparameter configuration with {the_model} and the optimization method: {the_optimization_method} is {best_hp.values}"
    )

    print(
        f"The percentage-RMS-difference for the configuration after the re-training equals to {evaluation}"
    )

    plt.plot(x_test.reshape(-1)[3000:4000], label="test-data")
    plt.plot(diff[3000:4000], label="reconstruction")
    plt.legend()
    plot("reconstruction")
