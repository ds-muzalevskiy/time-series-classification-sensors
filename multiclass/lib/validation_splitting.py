from sklearn.model_selection import StratifiedShuffleSplit


class StratifiedValidationSplitter:

    def __init__(self, test_size):
        self.test_size = test_size

    def split(self, X, y):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size)
        train_index, val_index = next(splitter.split(X, y))  # split returns a generator
        train_X, train_y = X[train_index], y[train_index]
        val_X, val_y = X[val_index], y[val_index]

        return train_X, train_y, val_X, val_y
