"""
Module to implement utility functions.
"""
import os
import csv
import pandas as pd

from sklearn.model_selection import train_test_split


def load_data(data_path):
    """
    Loads cleaned data from csv. Returns pandas DataFrame.
    """
    assert os.path.exists(data_path), "File does not exist"
    assert os.path.splitext(data_path)[1] == ".csv", "File is not a CSV"

    df = pd.read_csv(data_path)

    return df


def split_train_test(df, predictors, target, test_size=0.2):
    """
    Splits data in train, test and val sets. Returns dict of dataframes. 
    """

    assert target in df.columns.tolist(
    ), 'Target {self._target} not present in df.'
    assert all(elem in df.columns.tolist() for elem in predictors), ('Provided ' +
                                                    f'predictors {predictors} not present in df.')

    X = df[predictors]  # Features.
    y = df[target]  # Target variable.

    # Split the data into train and test sets with shuffling.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        shuffle=True, random_state=42)

    return {'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test}


def save_dict_to_csv(data_dict, data_path):
    """
    Save dict() to csv.
    """
    with open(data_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        writer.writeheader()
        writer.writerow(data_dict)
