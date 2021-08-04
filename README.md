# welcome to the hyper-parameter optimization land

In the repository we have implemented Random Search, Bayesian Optimiztion,
Hyperband and [HpBandSter](https://github.com/automl/HpBandSter) (BOHB).

## Setup

install [Poetry](https://python-poetry.org/docs/cli/) (`pip install poetry`) for
python 3.7.10;
Run `poetry install` to install the correct dependencies.
Poetry will create a `.venv` folder and put everything there.

## Get Dataset

Download the [Intel Data Lab Dataset](http://db.csail.mit.edu/labdata/data.txt.gz)
and set the "data_path" argument at [read_and_preprocess_data](https://github.com/ak4win/hyperparameter-optimization/blob/master/global_utils/get_data_multi_note.py) accordingly.



make sure you activate the virtualenv that was created by poetry beforehand via
this command:
```bash
source $(poetry env info --path)/bin/activate
```

## Run an experiment

After configuring the desired model and optimization algorithm in `optimization.py`, lean back and
run the code via this command (note that some of the algorithms may take very
long to finish (particularly so with the RNN as it isn't able to paralellize)):
```bash
python optimization.py
```
