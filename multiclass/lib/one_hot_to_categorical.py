import numpy as np


def one_hot_score_to_categorical(one_hot_array):
    """
    transforms a classification score that is one-hot encoded into a categorical result by assigning it the integer with
    the highest score.
    :param one_hot_array: a numpy array of shape (n, k) where n is the number of samples and k the number of categories.
    :return: a numpy array
    """

    return np.argmax(one_hot_array, axis=1)

