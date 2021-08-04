from optimization_utils.Optimization_Keras_Tuner.rnn import create_model_rnn
from optimization_utils.Optimization_Keras_Tuner.c_vae_model import create_model_c_vae
from optimization_utils.Optimization_HpBandSter.optimization_cbn_vae import run_experiments as run_experiments_cbn
from optimization_utils.Optimization_HpBandSter.optimization_rnn import run_experiments as run_experiments_rnn
model_configs = {
    "RNN": {
        "batch_size": 1,
        "sequence_length": 20,
        "train_test_cutoff": 1900,

        "optimization_epochs": 1,

        "retrain_epochs": 4,
        "should_early_stop": True,
        "retrain_early_stop_patience": 3,
        "rolling_avg_smoothing_window": 3,

        "create_model_fn": create_model_rnn,
        "hpbandster_run_experiments_function": run_experiments_rnn
    },
    "CBN_VAE": {
        "batch_size": 32,
        "sequence_length": 120,
        "train_test_cutoff": 320,

        "optimization_epochs": 32,

        "retrain_epochs": 500,
        "should_early_stop": False,
        "retrain_early_stop_patience": 35,
        "rolling_avg_smoothing_window": 5,  # 20 -> 9.731141667502332, 30 -> 10.51607613746664, 10 -> 9.454490065190768

        "create_model_fn": create_model_c_vae,
        "hpbandster_run_experiments_function": run_experiments_cbn
    }
}
