import pandas as pd
import os

from models import Model
from sklearn.model_selection import train_test_split


class BinaryClassification(Model):
    """
    Class that implements all methods for training, optimization, evaluation and prediction
    using ML models defined in Model for binary classification with numerical attributes.
    """
    def __init__ (self, model_name, target, predictors,
                  optimization=False, **kwargs):
        super().__init__(model_name, optimization, **kwargs)

        self._target = target
        self._predictors = predictors

    def _load_data(self, data_path):
        """
        Loads cleaned data from csv. Returns pandas DataFrame.
        """
        assert os.path.exists(data_path), "File does not exist"
        assert os.path.splitext(data_path)[1] == ".csv", "File is not a CSV"

        df = pd.read_csv(data_path)

        assert self._target in df.columns.tolist(), 'Target {self._target} not present in df'
        assert self._predictors in df.columns.tolist(), f'Provided predictors {self._predictors} \
                                                          not present in df'
        return df

    def split_train_test_val(self, data_path, test_size = 0.15, val_size = 0.15):
        """
        Splits data in train and test sets.
        """
        df = self._load_data(data_path)

        X = df[self._predictors] # Features.
        y = df[self._target] # Target variable.

        # Split the data into train and test sets with shuffling.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            shuffle=True, random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)

        return X_train, X_test, X_val,  y_train, y_test, y_val

    def train_model(self, model, X_train, y_train, val_data, callbacks,
                    batch_size=32, epochs=100, verbose=1):
        """
        Function to train model and store its weights. 
        """

        model.fit(X_train, y_train, validation_data=val_data,
                  batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose)
        # TODO: use cross-val (?), optimization here? callbacks