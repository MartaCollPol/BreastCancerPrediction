'''
File made for functionality testing. 
'''
from utils import utils
from binary_classification import classification as bc
from binary_classification.models import Model


if __name__ == "__main__":

    DATA_PATH = "data/clean_breast_cancer.csv"
    OUTPUT_PATH = "data/outputs"
    TARGET = 'diagnosis'
    PREDICTORS = ['radius_mean', 'texture_mean','perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean']

    # Load and split data.
    df = utils.load_data(DATA_PATH)
    data = utils.split_train_test(df, PREDICTORS, TARGET)

    #model_obj = Model('LogisticRegression', optimization=True)

    # Ensamble voting classifier
    estimators = [{'name':'LogisticRegression', 'params':{}},
                  {'name': 'Svm', 'params': {}},
                  {'name':'DecisionTree', 'params':{}}]
    model_obj = Model('EnsambleVoting', estimators=estimators, optimization=True)

    # Hyperparameter optimization.
    best_params = bc.hyperparameter_optimization(model_obj.model,
                                                model_obj.hyperparams_to_optimize,
                                                df[PREDICTORS],
                                                df[TARGET],
                                                cv=5)

    # Study with best params.
    # model_obj = Model('LogisticRegression', kwargs=best_params) 
    # # TODO: Check how best params is returned and how to launch voting classifier.

    bc.plot_learning_curve(model_obj, df[PREDICTORS], df[TARGET], cv=5)
    bc.train_and_evaluate_model(model_obj, data, save_outputs=True, output_dir=OUTPUT_PATH)
