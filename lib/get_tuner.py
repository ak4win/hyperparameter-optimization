from keras_tuner import RandomSearch, BayesianOptimization, Hyperband


def get_tuner(optimization_name, create_model_function, current_model):
    return {
        "RS": RandomSearch(
            create_model_function,  # model instance, whose hyper-parameters are optimized
            objective="val_loss",  # the direction of the optimization
            max_trials=50,  # the max. amount of model configurations that are tested
            executions_per_trial=1,  # how many rounds the model with that configuration is trained to reduce variance
            # seed=1,  # set the seed
            overwrite=True,  # boolean whether to overwrite the project
            directory=f"save_results/{current_model}/random_search",  # the relative path to the working directory
            project_name="results",
        ),
        "BO": BayesianOptimization(
            create_model_function,  # model instance, whose hyper-parameters are optimized
            objective="val_loss",  # the direction of the optimization
            max_trials=50,  # the max. amount of model configurations that are tested
            num_initial_points=3,  # number of randomly generated samples
            alpha=1e-4,  # the expected amount of noise in the observed performances
            beta=2.6,  # factor to balance exploration and explotation
            executions_per_trial=2,  # how many rounds the model with that configuration is trained to reduce variance
            # seed=1,  # set the seed
            overwrite=True,  # boolean whether to overwrite the project
            directory=f"save_results/{current_model}/bayesian",  # the relative path to the working directory
            project_name="results",
        ),
        "HB": Hyperband(
            create_model_function,  # model instance, whose hyper-parameters are optimized
            objective="val_loss",  # the direction of the optimization
            max_epochs=30,  # the max. amount of model configurations that are tested
            factor=4,  # reduction factor for the number of epochs and number of models for each bracket
            hyperband_iterations=6,  # number of times to iterate over the full Hyperband algorithm
            executions_per_trial=2,  # how many rounds the model with that configuration is trained to reduce variance
            # seed=1,  # set the seed
            overwrite=True,  # boolean whether to overwrite the project
            directory=f"save_results/{current_model}/hyperband",
            project_name="results",
        ),
    }[optimization_name]
