"""
Implements all methods for training, optimization, evaluation and prediction
using ML models defined in Model for binary classification with numerical attributes.
"""
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn import metrics
from models import Model


def _load_data(data_path):
    """
    Loads cleaned data from csv. Returns pandas DataFrame.
    """
    assert os.path.exists(data_path), "File does not exist"
    assert os.path.splitext(data_path)[1] == ".csv", "File is not a CSV"

    df = pd.read_csv(data_path)

    return df


def plot_learning_curve(model_obj, X, y, cv=None,
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plots learning curve of a model training to asses if it is
    underfitting or overfitting using cross-validation.
    """
    model = model_obj.model

    train_sizes, train_scores, val_scores = learning_curve(model,
                                                           X,
                                                           y,
                                                           cv=cv,
                                                           train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation Score')

    plt.legend(loc='best')
    plt.show()


def split_train_test_val( data_path, predictors, target, test_size = 0.15, val_size = 0.15):
    """
    Splits data in train, test and val sets. Returns dict of dataframes. 
    """
    df = _load_data(data_path)

    assert target in df.columns.tolist(), 'Target {self._target} not present in df.'
    assert predictors in df.columns.tolist(), f'Provided predictors {predictors} \
                                                        not present in df.'

    X = df[predictors] # Features.
    y = df[target] # Target variable.

    # Split the data into train and test sets with shuffling.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        shuffle=True, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)

    return {'X_train': X_train, 'X_test': X_test, 'X_val': X_val,
            'y_train': y_train, 'y_test': y_test, 'y_val': y_val}


def evaluation(y_true, y_pred):
    """
    Given target real values and predicted ones by a model, this function
    calculates different classification metrics and return them in a dict.
    """
    results = {}

    results['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    results['precision'] = metrics.precision_score(y_true, y_pred)
    results['recall'] = metrics.recall_score(y_true, y_pred)
    results['f1'] = metrics.f1_score(y_true, y_pred)
    results['auc_roc'] = metrics.roc_auc_score(y_true, y_pred)
    results['confusion'] = metrics.confusion_matrix(y_true, y_pred)

    return results


def train_and_evaluate_model(model_obj, data, batch_size=32, epochs=10,
                             verbose=1, store_outputs=False):
    """
    Function to train model and store its weights. 
    """
    assert isinstance(model_obj, Model), "model_obj has to be of class Model."

    model = model_obj.model
    model.fit(data.X_train, data.y_train, validation_data=(data.X_val, data.y_val),
                batch_size=batch_size, epochs=epochs, verbose=verbose)

    preds_train = model.predict(data.X_train)
    preds_test = model.predict(data.X_test)

    eval_train = pd.DataFrame(evaluation(data.y_train, preds_train))
    eval_test = pd.DataFrame(evaluation(data.y_test, preds_test))

    print(f"For model {model_obj.model_name} with {model_obj.model_hyperparameters}:\n")
    print("Results for train set:\n")
    print(eval_train)
    print("\nResults for test set:\n")
    print(eval_test)

    if store_outputs:
        # Save the model
        with open(f'data/outputs/{model_obj.model_name}/model.pkl', 'wb') as file:
            pickle.dump(model, file)

        # Save the evaluation
        eval_train.to_csv(f'data/outputs/{model_obj.model_name}/Train_Evaluation.csv')
        eval_test.to_csv(f'data/outputs/{model_obj.model_name}/Test_Evaluation.csv')

# TODO: hyperparam optimization method