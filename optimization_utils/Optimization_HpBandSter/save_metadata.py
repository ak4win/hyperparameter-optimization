import json
from datetime import timedelta


def save_metadata(current_model: str, current_optimization_method: str, optimization_time: timedelta,
                  retraining_time: timedelta, num_configs: int) -> None:
    with open(f'save_results/{current_model}/{current_optimization_method}/results.json', 'r') as file:
        tried_configs_raw = file.readlines()
    with open(f'save_results/{current_model}/{current_optimization_method}/results.json', 'w') as file:
        tried_configs = list(map(lambda x: json.loads(x), tried_configs_raw))
        file.write(json.dumps({
            "tried_configs": tried_configs,
            "meta_data": {
                "optimization_time": str(optimization_time),
                "tried_configs": num_configs,
                "retraining_time": str(retraining_time)
            }
        }))
