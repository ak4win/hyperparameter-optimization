# welcome to the hyper-parameter optimization land

In the repository we have implemented Random Search, Bayesian Optimiztion, Hyperband and [HpBandSter](https://github.com/automl/HpBandSter).

## relax and enjoy the optimization for your neural network but first things first, the setup:

install [Poetry](https://python-poetry.org/docs/cli/):
create and activate a virtual environment using python 3.7.10 as defined in the pyproject.toml
run poetry install to install dependencies into your environment

## Get Dataset

Download the [Intel Data Lab Dataset](http://db.csail.mit.edu/labdata/data.txt.gz)
and set the "data_path" argument at read_and_preprocess_data accordingly.

## run an experiment

set the variables you want at the [optimization.py](https://github.com/ak4win/hyperparameter-optimization/blob/master/optimization.py) file and run the experiment by entering by runing optimization.py with the command
python optimization.py
