import pandas as pd
import numpy as np

import json
import os
from pathlib import Path, WindowsPath

import re


class DataReader:
    """
    To recognize training and test set in a given directory and read them in
    """

    TRAIN_REGEX = re.compile("train.*\.csv")
    TEST_REGEX = re.compile("test.*\.csv")

    ID_COLUMN = "id"
    CATEGORY_COLUMN = 'category'
    TS_COLUMN = 'ts'

    def __init__(self, folder):

        self.folder = Path(folder)
        self.train_file = self.folder / self.get_train_name()
        self.test_file = self.folder / self.get_test_name()
        self.train_df = None
        self.test_df = None

    def _get_filenames_that_match_regex(self, a_regex):
        all_files_in_folder = os.listdir(self.folder)
        return [f for f in all_files_in_folder if a_regex.match(f)]

    def _get_one_match_or_fail(self, a_regex):
        matching_file_list = self._get_filenames_that_match_regex(a_regex)
        DataReader._raise_if_not_len_one(matching_file_list)
        return matching_file_list[0]

    @staticmethod
    def _raise_if_not_len_one(a_list):
        list_length = len(a_list)
        if list_length is not 1:
            raise ValueError("List should contain one element but found {} in {}".format(list_length, a_list))

    def get_train_name(self):
        return self._get_one_match_or_fail(self.TRAIN_REGEX)

    def get_test_name(self):
        return self._get_one_match_or_fail(self.TEST_REGEX)

    def set_train_data_set(self):
        self.train_df = pd.read_csv(self.train_file)

    def set_test_data_set(self):
        self.test_df = pd.read_csv(self.test_file)

    def set_data_sets(self):
        self.set_train_data_set()
        self.set_test_data_set()

    def get_data_sets(self):
        """
        To be used by the end user.
        :return: tuple containing training and test data set
        """

        if self.train_df is None and self.test_df is None:
            self.set_data_sets()

        return self.train_df, self.test_df
