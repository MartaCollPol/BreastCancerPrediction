import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def load_data(data_path):
    """
    Loads cleaned data from csv. Returns pandas DataFrame.
    """
    assert os.path.exists(data_path), "File does not exist"
    assert os.path.splitext(data_path)[1] == ".csv", "File is not a CSV"

    df = pd.read_csv(data_path)

    return df


def split_train_test(df, predictors, target, test_size = 0.2):
    """
    Splits data in train, test and val sets. Returns dict of dataframes. 
    """

    assert target in df.columns.tolist(), 'Target {self._target} not present in df.'
    assert all(elem in df.columns.tolist() for elem in predictors), (f'Provided ' + 
                                        'predictors {predictors} not present in df.')

    X = df[predictors] # Features.
    y = df[target] # Target variable.

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


def save_model_outputs(model, model_name, hyperparams, eval_train, eval_test, output_dir):
    """
    Saves model to a pickle, model hyperparams and evaluation results to output_dir.
    """
    # Create directories if they don't exist.
    model_dir = output_dir + '/' + model_name
    os.makedirs(model_dir, exist_ok=True)

    # Save the model.
    with open(f'{model_dir}/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Save best params.
    utils.save_dict_to_csv(hyperparams, f'{model_dir}/params.csv' )

    # Save the evaluation.
    eval_train.to_csv(f'{model_dir}/Train_Evaluation.csv')
    eval_test.to_csv(f'{model_dir}/Test_Evaluation.csv')


def plot_confusion_matrix(confusion_matrix, class_labels):
    """
    Plots a confusion matrix using a headmap.
    """
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)

    # Add labels, title, and ticks
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Test set')
    plt.xticks(np.arange(len(class_labels)), class_labels)
    plt.yticks(np.arange(len(class_labels)), class_labels)
