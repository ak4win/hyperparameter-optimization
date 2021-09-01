import tensorflow as tf
from typing import Any


def mean_squared_error_without_nans(y_true: Any, y_pred: Any) -> Any:
    residuals = tf.square(y_true - y_pred)
    residuals_no_nan = tf.where(tf.math.is_nan(residuals), tf.fill(residuals.shape, float(10_000)), residuals)
    sum_residuals = tf.reduce_mean(residuals_no_nan)

    return tf.cast(sum_residuals, dtype=tf.float32) 


if __name__ == '__main__':
    one = tf.ones((1,))
    nan = tf.constant(float('nan'))
    print(mean_squared_error_without_nans(one, nan).dtype)
