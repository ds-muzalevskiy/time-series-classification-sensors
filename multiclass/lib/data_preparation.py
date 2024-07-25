import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Transformer:
    """
    Abstract class implementing the same interface as sklearn methods. Transformers have to be fit once and can then be
    used with transform on similar data.
    """

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, data):
        pass

    def transform(self, data):
        return data


class LengthChanger(Transformer):
    """
    Abstract class for transformers that change only the length of the time series.
    """

    def __init__(self, id_column):
        self.id_column = id_column
        self.transform_function = None

    def fit(self, data):
        self.transform_function = self._make_transform_function(data=data)

    def transform(self, data):
        transformed_data = data.groupby([self.id_column], as_index=False, sort=False).apply(self.transform_function)
        # unfortunately index behavior after grouping is hard to predict, therefore I return just standard index
        return transformed_data.reset_index(drop=True)

    def _make_transform_function(self, data):
        return lambda x: x

    def _get_longest_time_series(self, a_df):
        return a_df.groupby(self.id_column).size().max()


class LengthTruncater(LengthChanger):
    """
    Cuts down length of time series to the maximum length it has seen during fit.
    """

    def _make_transform_function(self, data):

        max_length = self._get_longest_time_series(data)
        return self._get_truncate_function(max_length)

    def _get_truncate_function(self, max_length):

        def transform_function(a_df):

            if len(a_df) > max_length:
                return a_df.iloc[-max_length:, :]
            else:
                return a_df

        return transform_function


class ZeroPadder(LengthChanger):
    """
    Pads time series with zeros to length of longest time series it has seen during fit.
    """

    def __init__(self, id_column, cols_not_to_pad):
        super(ZeroPadder, self).__init__(id_column=id_column)
        self.cols_not_to_pad = cols_not_to_pad
        if self.id_column not in cols_not_to_pad:
            self.cols_not_to_pad.append(self.id_column)

    def _make_transform_function(self, data):

        max_length = self._get_longest_time_series(data)

        return self._get_padding_function(min_length=max_length)

    def _get_padding_function(self, min_length):

        cols_not_to_pad = self.cols_not_to_pad

        def transform_function(a_df):

            n_missing_lines = min_length - len(a_df)

            if n_missing_lines > 0:
                transformed_data = ZeroPadder.pad_k_lines(data=a_df, how_many=n_missing_lines, cols_not_to_pad=cols_not_to_pad)
            else:
                transformed_data = a_df
            return transformed_data
        return transform_function

    @staticmethod
    def pad_k_lines(data, how_many, cols_not_to_pad):

        data_columns = data.columns

        vals_to_pad = {
            col: data[col].values[0] if col in cols_not_to_pad else 0
            for col in data_columns
        }

        pad_data = {
            key: [value]*how_many for key, value in vals_to_pad.items()
        }
        new_lines = pd.DataFrame(data=pad_data)
        return pd.concat([new_lines, data])


class ColsDeleter(Transformer):
    """
    Deletes columns by name.
    """

    def __init__(self, cols_to_del):
        self.cols_to_del = cols_to_del

    def transform(self, data):
        return data.drop(self.cols_to_del, axis=1)


class To3dNumpyConverter(Transformer):
    """
    Converts data frame representing a multivariate time series to a 3d numpy array.
    """

    def __init__(self, id_column):
        self.id_column = id_column

    def transform(self, data):
        list_of_2d_arrays = [self._df_to_2d(multivariate_ts)
                             for multivariate_ts in self._get_list_of_ts_values(data)]
        return np.stack(list_of_2d_arrays, axis=0)

    def _get_list_of_ts_values(self, a_df):
        return [multivariate_ts for _, multivariate_ts in a_df.groupby(self.id_column, sort=False)]

    def _df_to_2d(self, df):
        multi_variate_ts_values = df.drop(self.id_column, axis=1).values
        return multi_variate_ts_values


class FeatureLabelSplitter(Transformer):
    """
    Splits data into a data frame containing only the features and a dataframe containing only the labels.
    """

    def __init__(self, id_column, label_column):
        self.id_column = id_column
        self.label_column = label_column

    def transform(self, data):

        label_df = data.loc[:, [self.id_column, self.label_column]]
        feature_df = data.drop(self.label_column, axis=1)

        return feature_df, label_df


class LabelCondenser(Transformer):
    """
    Condenses a data frame representing labels to a single value.
    """

    def __init__(self, id_column, label_column):
        self.id_column = id_column
        self.label_column = label_column

    def transform(self, data):
        labels_by_id = data.groupby(self.id_column, sort=False).apply(self._condense)
        return labels_by_id.values.reshape(-1, 1)

    def _condense(self, df_chunk):
        return df_chunk[self.label_column].values[0]


class XyTransformer(Transformer):
    """
    Applies different Transformers to features and labels.
    """

    def __init__(self, x_transformer=None, y_transformer=None):
        self.x_transformer = x_transformer or Transformer()
        self.y_transformer = y_transformer or Transformer()

    @staticmethod
    def _check_input_correct(data):
        if len(data) is not 2 or not isinstance(data, tuple):
            raise ValueError("Data must be tuple of length 2, found %s" % data)

    def transform(self, data):
        XyTransformer._check_input_correct(data)
        x, y = data
        transformed_x = self.x_transformer.transform(x)
        transformed_y = self.y_transformer.transform(y)
        return transformed_x, transformed_y

    def fit(self, data):
        XyTransformer._check_input_correct(data)
        x, y = data
        self.x_transformer.fit(x)
        self.y_transformer.fit(y)


class TransformerPipeline(Transformer):
    """
    Plugs together different Transformers.
    """

    def __init__(self, transformer_list):

        self.transformer_list = transformer_list

    def fit_transform(self, data):

        for transformer in self.transformer_list:
            transformer.fit(data)
            data = transformer.transform(data)

        return data

    def fit(self, data):
        # cannot actually do fit of the transformers down the line without actually transforming
        self.fit_transform(data)

    def transform(self, data):

        for transformer in self.transformer_list:
            data = transformer.transform(data)
        return data

    def append(self, a_transformer):
        self.transformer_list.append(a_transformer)

    def prepend(self, a_transformer):
        self.transformer_list.insert(0, a_transformer)


class XyMaker(TransformerPipeline):
    """
    Makes suitable features and labels out of a dataframe with multivariate time series of equal length.
    """

    def __init__(self, id_column, label_column):
        transformer_list = self.get_transformer_list(id_column=id_column, label_column=label_column)
        super(XyMaker, self).__init__(transformer_list)


    @staticmethod
    def get_transformer_list(id_column, label_column):
        return [
            FeatureLabelSplitter(id_column=id_column, label_column=label_column),
            XyTransformer(
                x_transformer=To3dNumpyConverter(id_column=id_column),
                y_transformer=LabelCondenser(id_column=id_column, label_column=label_column),
            )
        ]


class TensorflowTsDataPreparer(TransformerPipeline):
    """
    Prepares multivariate data for Tensorflow by cutting unnecessary time column, bringing to same length, splitting
    into labels and features and converting to numpy arrays.
    """

    def __init__(self, id_column, label_column, ts_column):

        transformer_list = TensorflowTsDataPreparer.get_transformer_list(id_column=id_column,
                                                                         label_column=label_column,
                                                                         ts_column=ts_column)
        super(TensorflowTsDataPreparer, self).__init__(transformer_list)

    @staticmethod
    def get_transformer_list(id_column, label_column, ts_column):

        return [
            ColsDeleter(cols_to_del=[ts_column]),
            LengthTruncater(id_column=id_column),
            DataFrameScaler(scaler=StandardScaler(), cols_to_ignore=[id_column, label_column]),
            ZeroPadder(id_column=id_column, cols_not_to_pad=[label_column]),
            XyMaker(id_column=id_column, label_column=label_column),
            XyTransformer(y_transformer=CategoriesToOneHot()),

        ]


class DataFrameScaler(Transformer):
    """
    Applies scalers to certain columns in the data.
    """

    def __init__(self, scaler, cols_to_ignore):
        """
        :param scaler: a scaler from sklearn.preprocessing
        :param cols_to_ignore:
        """
        self.cols_to_ignore = cols_to_ignore
        self.cols_to_scale = []
        self.scaler = scaler

    def _set_cols_to_scale(self, data):
        self.cols_to_scale =  [col for col in data.columns if col not in self.cols_to_ignore]

    def _get_data_to_scale(self, data):
        cols_to_scale = self.cols_to_scale
        data_for_scaling = data.loc[:, cols_to_scale].values
        return data_for_scaling

    def fit(self, data):
        self._set_cols_to_scale(data)
        data_for_scaling = self._get_data_to_scale(data)
        self.scaler.fit(data_for_scaling)

    def transform(self, data):

        data_for_scaling = self._get_data_to_scale(data)
        scaled_data = self.scaler.transform(data_for_scaling)
        return self.insert_scaled_data(scaled_data, data)

    def insert_scaled_data(self, scaled_data, data):

        data.loc[:, self.cols_to_scale] = scaled_data
        return data


class NumpyReshaper(Transformer):
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def transform(self, data):
        return data.reshape(self.new_shape)


class CategoriesToOneHot(TransformerPipeline):

    def __init__(self):
        transformer_list = CategoriesToOneHot.get_transformer_list()
        super(CategoriesToOneHot, self).__init__(transformer_list)

    @staticmethod
    def get_transformer_list():
        return [
            # Reshaping because LabelEncoder wants a row vector
            NumpyReshaper(new_shape=(-1, )),
            LabelEncoder(),
            # Reshaping because NumpyReshaper wants column vector
            NumpyReshaper(new_shape=(-1, 1)),
            OneHotEncoder(sparse=False),
        ]

