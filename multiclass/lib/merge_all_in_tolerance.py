import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def get_dummy_key(left, right):
    key = "DUMMYKEY"
    suffix = 'X'

    while (key in left.columns) or (key in right.columns):
        key += suffix

    return key


def add_dummy_merge_key(left_df, right_df, dummy_key):
    left_df, right_df = left_df.copy(), right_df.copy()

    left_df[dummy_key] = 0
    right_df[dummy_key] = 0

    return left_df, right_df


def merge_all_in_tolerance(left, right, left_on, right_on, tolerance):

    dummy_key = get_dummy_key(left, right)

    left, right = add_dummy_merge_key(left, right, dummy_key)

    left_first_line = left.iloc[:1, :]

    left_value = left_first_line[left_on].values[0]

    match_right = get_time_before_up_to_tolerance(left_value, right, right_on, tolerance)

    merged_df = left_first_line.merge(match_right, how='inner', on=dummy_key)
    merged_df = merged_df.drop([dummy_key], axis=1)
    
    return merged_df


def iterate_over_rows(a_df):
    for i in range(len(a_df)):
        yield a_df.iloc[[i], :]


def get_time_before_up_to_tolerance(left_value, right_df, right_on, tolerance):

    right_time = right_df[right_on]

    before = right_time <= left_value
    within_tolerance = (left_value - right_time) < tolerance

    return right_df[before & within_tolerance]


def merge_asof_outer(left, right, left_on, right_on, tolerance):

    def func(left_df):
        return merge_all_in_tolerance(left_df, right, left_on, right_on, tolerance)

    return pd.concat([func(df) for df in iterate_over_rows(left)])
