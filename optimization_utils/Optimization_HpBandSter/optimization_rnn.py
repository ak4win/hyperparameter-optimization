"""
LSTM-RNN-Auto-Encoder
=================
Optimization of the LSTM-RNN Model which is implemented in our research project.
This example also shows how to log results to disk during the optimization
which is useful for long runs, because intermediate results are directly available
for analysis. It also contains a more realistic search space with different types
of variables to be optimized.
"""
import logging
from hpbandster.optimizers import BOHB
import hpbandster.core.nameserver as hpns
import pickle
import os
import tensorflow as tf
from global_utils.get_data_multi_note import read_and_preprocess_data
from optimization_utils.Optimization_HpBandSter.optimization_cbn_vae import (
    JsonResultsSaver,
)
from optimization_utils.Optimization_HpBandSter.create_arguments import create_arguments
from optimization_utils.Optimization_HpBandSter.worker_rnn import KerasWorker as worker
from numpy.random import seed

seed(1)

tf.random.set_seed(2)


logging.basicConfig(level=logging.DEBUG)


args = create_arguments(1, 8, "RNN")


def run_experiments(x_train, x_test, overwrite=False):
    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)

    if args.worker:
        import time

        time.sleep(
            5
        )  # short artificial delay to make sure the nameserver is already running
        w = worker(x_train, x_test, run_id=args.run_id, host=host, timeout=120)
        w.load_nameserver_credentials(working_directory=args.shared_directory)
        w.run(background=False)
        exit(0)

    # This example shows how to log live results. This is most useful
    # for really long runs, where intermediate results could already be
    # interesting. The core.result submodule contains the functionality to
    # read the two generated files (results.json and configs.json) and
    # create a Result object.
    result_logger = JsonResultsSaver(
        directory=args.shared_directory, overwrite=overwrite
    )

    # Start a nameserver:
    NS = hpns.NameServer(
        run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory
    )
    ns_host, ns_port = NS.start()

    # Start local worker
    w = worker(
        x_train,
        x_test,
        run_id=args.run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        timeout=120,
    )
    w.run(background=True)

    # Run an optimizer
    bohb = BOHB(
        configspace=worker.get_configspace(),
        run_id=args.run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        result_logger=result_logger,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
    )
    res = bohb.run(n_iterations=args.n_iterations)
    print(res)

    # store results
    with open(os.path.join(args.shared_directory, "results.pkl"), "wb") as fh:
        pickle.dump(res, fh)

    # shutdown
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()


if __name__ == "__main__":
    x_train, x_test = read_and_preprocess_data(
        sequence_length=20,
        batch_size=1,
        motes_train=[7],
        motes_test=[7],
    )
    x_train = x_train[:100, :, :]
    x_test = x_test[200:225, :, :]
    run_experiments(x_train, x_test, overwrite=True)
