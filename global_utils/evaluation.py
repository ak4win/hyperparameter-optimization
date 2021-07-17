###

# from preprocessing import *
import numpy as np

# Compression Ratio >> compressed data lenght over the size of uncompressed data

def compression_ratio(dic_name, batches):
    
    dictionary = dic_name
    len_of_one_batch = len(batches[0])

    hidden_dim = [i['hidden_dim'] for i in dictionary if 'hidden_dim' in i]
    smallest_hidden_dim = min(hidden_dim) ### + the size of the index matrix

    ratio = len_of_one_batch / smallest_hidden_dim

    return ratio

### ------------------------------------------------------------------------------------ ###
### ------------------------------------------------------------------------------------ ###
### ------------------------------------------------------------------------------------ ###

# Percentage RMS (Root-mean-square) difference
def per_rms_diff(label, prediction):
    # assert prediction.dtype == tf.float32, f'wrong dtype for prediction {prediction.dtype}'
    # assert label.dtype == tf.float32, f'wrong dtype for label {prediction.dtype}'
    try:
        diff = (np.sum(np.square(np.subtract(label, prediction))))
        sq = (np.sum(np.square(label)))
        prd = (100 * (np.sqrt(np.divide(diff, sq))))
    except ZeroDivisionError:
        print("Oh, no! You tried to divide by zero!")

    # Try:
    #     rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    # Except ZeroDivisionError:
    #     print("Oh, no! You tried to divide by zero!")

    # # prediction = tf.convert_to_tensor(prediction)
    # # label = tf.convert_to_tensor(label)

    # # diff = tf.square(tf.subtract(prediction, label))
    # # sq = tf.square(diff)
    # # rms_diff = tf.sqrt(tf.divide(diff, sq))

    # # prd = tf.cast(tf.reduce_mean(rms_diff), tf.float32)
    # # prd = prd * 100

    return prd