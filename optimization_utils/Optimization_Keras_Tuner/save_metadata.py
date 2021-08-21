
from datetime import timedelta
import json


def save_metadata(current_model: str, current_optimization_method: str, optimization_time: timedelta,
                  retraining_time: timedelta, num_configs: int) -> None:
    with open(f'save_results/{current_model}/{current_optimization_method}/results.json', 'w') as file:
        file.write(json.dumps({
            "meta_data": {
                "optimization_time": str(optimization_time),
                "tried_configs": num_configs,
                "retraining_time": str(retraining_time)
            }
        }))
