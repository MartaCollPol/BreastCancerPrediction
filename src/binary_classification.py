"""
Implements all methods for training, optimization, evaluation and prediction
using ML models defined in Model for binary classification with numerical attributes.
"""
import pandas as pd
import os
import pickle

from models import Model
from sklearn.model_selection import train_test_split
from sklearn import metrics


def _load_data(data_path):
    """
    Loads cleaned data from csv. Returns pandas DataFrame.
    """
    assert os.path.exists(data_path), "File does not exist"
    assert os.path.splitext(data_path)[1] == ".csv", "File is not a CSV"

    df = pd.read_csv(data_path)

    return df


def split_train_test_val( data_path, predictors, target, test_size = 0.15, val_size = 0.15):
    """
    Splits data in train and test sets.
    """
    df = _load_data(data_path)

    assert target in df.columns.tolist(), 'Target {self._target} not present in df'
    assert predictors in df.columns.tolist(), f'Provided predictors {predictors} \
                                                        not present in df'

    X = df[predictors] # Features.
    y = df[target] # Target variable.

    # Split the data into train and test sets with shuffling.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        shuffle=True, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)

    return X_train, X_test, X_val,  y_train, y_test, y_val


def evaluation(y_true, y_pred):
    """
    Given target real values and predicted ones by a model, this function
    calculates different classification metrics and return them in a dict.
    """
    results = dict()

    results['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    results['precision'] = metrics.precision_score(y_true, y_pred)
    results['recall'] = metrics.recall_score(y_true, y_pred)
    results['f1'] = metrics.f1_score(y_true, y_pred)
    results['auc_roc'] = metrics.roc_auc_score(y_true, y_pred)
    results['confusion'] = metrics.confusion_matrix(y_true, y_pred)

    return results


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, val_data,
                batch_size=32, epochs=10, verbose=1, store_outputs=False):
    """
    Function to train model and store its weights. 
    """

    model.fit(X_train, y_train, validation_data=val_data,
                batch_size=batch_size, epochs=epochs, verbose=verbose)
    
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    eval_train = pd.DataFrame(evaluation(y_train, preds_train))
    eval_test = pd.DataFrame(evaluation(y_test, preds_test))
    
    print(f"For model {model.model_name} with {model.model_hyperparameters}:\n")
    print(f"Results for train set:\n")
    print(eval_train)
    print(f"\nResults for test set:\n")
    print(eval_test)

    if (store_outputs):
        # Save the model
        with open(f'data/outputs/{model.model_name}/model.pkl', 'wb') as file:
            pickle.dump(model, file)

        # Save the evaluation
        eval_train.to_csv(f'data/outputs/{model.model_name}/Train_Evaluation.csv')
        eval_test.to_csv(f'data/outputs/{model.model_name}/Test_Evaluation.csv')

    # TODO: use cross-val (?), optimization, normalize