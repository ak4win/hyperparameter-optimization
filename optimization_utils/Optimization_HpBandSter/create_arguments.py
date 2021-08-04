import argparse


def create_arguments(min_budget, max_budget, model_name):
    parser = argparse.ArgumentParser(
        description=f"{model_name} model of our research project is optimized by the HpBandSter Methdod."
    )
    parser.add_argument(
        "--min_budget",
        type=float,
        help="Minimum number of epochs for training.",
        default=min_budget,
    )
    parser.add_argument(
        "--max_budget",
        type=float,
        help="Maximum number of epochs for training.",
        default=max_budget,
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        help="Number of iterations performed by the optimizer",
        default=16,
    )
    parser.add_argument(
        "--worker", help="Flag to turn this into a worker process", action="store_true"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler",
    )
    parser.add_argument(
        "--nic_name",
        type=str,
        help="Which network interface to use for communication.",
        default="lo",
    )
    parser.add_argument(
        "--shared_directory",
        type=str,
        help="A directory that is accessible for all processes, e.g. a NFS share.",
        default=f"save_results/{model_name}/HpBandSter",
    )
    parser.add_argument(
        "--backend",
        help="Toggles which worker is used. Choose between a pytorch and a keras implementation.",
        choices=["pytorch", "keras"],
        default="keras",
    )

    args = parser.parse_args()

    return args
