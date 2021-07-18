###

# from preprocessing import *
import numpy as np
import pandas as pd

# Compression Ratio >> compressed data lenght over the size of uncompressed data


def compression_ratio(dic_name, batches):

    dictionary = dic_name
    len_of_one_batch = len(batches[0])

    hidden_dim = [i["hidden_dim"] for i in dictionary if "hidden_dim" in i]
    smallest_hidden_dim = min(hidden_dim)  ### + the size of the index matrix

    ratio = len_of_one_batch / smallest_hidden_dim

    return ratio


### ------------------------------------------------------------------------------------ ###
### ------------------------------------------------------------------------------------ ###
### ------------------------------------------------------------------------------------ ###

# Percentage RMS (Root-mean-square) difference
def per_rms_diff(label, prediction):
    try:
        diff = np.sum(np.square(np.subtract(label, prediction)))
        sq = np.sum(np.square(label))
        prd = 100 * (np.sqrt(np.divide(diff, sq)))
    except ZeroDivisionError:
        print("Oh, no! You tried to divide by zero!")

    return prd


### ------------------------------------------------------------------------------------ ###
### ------------------------------------------------------------------------------------ ###
### ------------------------------------------------------------------------------------ ###

# smooth predictions for rolling mean
def smooth_output(test_data, predictions, smoothing_window):
    reshape_predictions = predictions.reshape(-1)
    smooth_predictions = pd.Series(reshape_predictions)
    smooth_predictions = smooth_predictions.rolling(window=smoothing_window).mean()[
        smoothing_window:
    ]
    smooth_output = smooth_predictions.to_numpy()
    diff = per_rms_diff(
        test_data.reshape(-1)[smoothing_window:], smooth_output.reshape(-1)
    )

    return smooth_output, diff
