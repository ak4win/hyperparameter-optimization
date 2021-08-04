import os
import json


def get_loss_of_result(result):
    try:
        res = result['result']['loss']
    except Exception:
        res = float('inf')
    return res


def get_best_result(current_model):
    path = os.path.join(os.getcwd(), f'save_results/{current_model}/HpBandSter/results.jsonl')
    with open(path, 'r') as file:
        results_raw = file.read().splitlines()
    results = list(map(lambda x: json.loads(x), results_raw))

    results.sort(key=get_loss_of_result)
    best_config = results[0]['config']

    if current_model == 'RNN':
        from lib.create_rnn import create_model
    else:
        from lib.create_cbn_vae import create_model

    return best_config, create_model(best_config)


if __name__ == '__main__':
    get_best_result('CBN_VAE')
