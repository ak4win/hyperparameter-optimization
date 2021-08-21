def get_num_configs(current_model: str, current_optimiziation: str) -> int:
    with open(f'save_results/{current_model}/{current_optimiziation}/results.json', 'r') as file:
        lines = file.readlines()
    return len(lines)
