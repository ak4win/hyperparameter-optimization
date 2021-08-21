from datetime import timedelta

from optimization_utils.Optimization_HpBandSter.save_metadata import save_metadata as save_bohb
from optimization_utils.Optimization_Keras_Tuner.save_metadata import save_metadata as save_tuner


def save_metadata(current_model: str, current_optimization_method: str, optimization_time: timedelta,
                  retraining_time: timedelta, num_configs: int) -> None:
    saving_fn = save_bohb if current_optimization_method == 'HpBandSter' else save_tuner
    saving_fn(current_model, current_optimization_method, optimization_time, retraining_time, num_configs)
