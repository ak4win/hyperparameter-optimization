#

# from preprocessing import *
import numpy as np
import pandas as pd

# Compression Ratio >> compressed data lenght over the size of uncompressed data


def compression_ratio(dic_name, batches):

    dictionary = dic_name
    len_of_one_batch = len(batches[0])

    hidden_dim = [i["hidden_dim"] for i in dictionary if "hidden_dim" in i]
    smallest_hidden_dim = min(hidden_dim)  # + the size of the index matrix

    ratio = len_of_one_batch / smallest_hidden_dim

    return ratio


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# Percentage RMS (Root-mean-square) difference
def per_rms_diff(label, prediction):
    try:
        diff = np.sum(np.square(np.subtract(label, prediction)))
        sq = np.sum(np.square(label))
        prd = 100 * (np.sqrt(np.divide(diff, sq)))
    except ZeroDivisionError:
        print("Oh, no! You tried to divide by zero!")

    return prd


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# smooth predictions for rolling mean
def smooth_output(test_data, predictions, smoothing_window):
    original_shape = predictions.shape
    reshaped_predictions = predictions.reshape(-1)

    # Smooth the values
    smoothed_predictions = pd.Series(reshaped_predictions)
    smoothed_predictions = smoothed_predictions.rolling(window=smoothing_window).mean()[smoothing_window:]
    smoothed_predictions = smoothed_predictions.to_numpy()

    # Put back to original shape
    smoothed_predictions = np.pad(smoothed_predictions, (smoothing_window, 0), 'constant')
    smoothed_predictions = smoothed_predictions.reshape(original_shape)

    diff = per_rms_diff(test_data.reshape(-1), smoothed_predictions.reshape(-1))
    return smoothed_predictions, diff
