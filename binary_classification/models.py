from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB # Asumes features are independent
                                           # and follow a gaussian distribution.


class Model():
    """
    Class for model initialization and model properties.
    """
    def __init__(self, model_name, optimization=False, **kwargs):
        self._model_name = model_name
        self._optimization = optimization
        self._hyperparameters = None
        self._default_hyperparams_to_optimize = None
        self._model_list = ['LogisticRegression',
                            'RandomForest', 'DecisionTree', 'SVM',
                            'NaiveBayes']
        self._model = self.define_model(**kwargs)

        assert self._model_name in self._model_list, (f"Model {model_name} is not available," +
                                                      f"possible models are {self._model_list}")

    @property
    def model_name(self):
        """
        Returns Model name.
        """
        return self._model_name

    @property
    def available_models(self):
        """
        Returns available model options.
        """
        return self._model_list

    @property
    def hyperparameters(self):
        """
        Returns model hyperparameters used in model definition (dict).
        """
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, new_value):
        """
        Sets a new value for model hyperparameter.
        """
        self._hyperparameters = new_value

    @property
    def hyperparams_to_optimize(self):
        """
        Returns hyperparameters' ranges to explore in hyperparameter optimization
        for defined model (dict).
        """
        return self._default_hyperparams_to_optimize

    @hyperparams_to_optimize.setter
    def hyperparams_to_optimize(self, new_value):
        """
        Returns hyperparameters' ranges to explore in hyperparameter optimization
        for defined model (dict).
        """
        self._default_hyperparams_to_optimize = new_value

    @property
    def model(self):
        """
        Defines and returns a model based on initialization class parameters.
        """
        return self._model


    def define_model(self, **kwargs):
        """
        Defines a model given it's name using default hyperparameters if not given.
        Returns defined model and hyperparameters used.
        """

        if self.model_name == 'LogisticRegression':
            # Return model initialization without hyperparams if optimization = True.
            if self._optimization:
                # Default hyperparameter ranges to explore in hyperparameter optimization.
                self.hyperparams_to_optimize = {'penalty': ['l2'],
                                                'C': [100, 1000, 10000],
                                                'solver': ['newton-cg',
                                                           'lbfgs',
                                                           'liblinear',
                                                           'sag',
                                                           'saga'],
                                                'max_iter': [5000, 6000, 7000]}                
                return LogisticRegression()

            # Hyperparameters to be used when defining the model.
            penalty = kwargs.get('penalty', 'l2') # Regularization penalty term.
            c_reg = kwargs.get('C', 1.0) # Inverse of regularization strength.
            solver = kwargs.get('solver', 'lbfgs') # Algorithm to use for optimization.
            max_iter = kwargs.get('max_iter',
                                  100) # Maximum number of iterations for the solver to converge.

            # Hyperparameters that will be used when defining the model.
            self.hyperparameters = {'penalty': penalty,
                                    'C': c_reg,
                                    'solver': solver,
                                    'max_iter': max_iter}

            # Model definition.
            model = LogisticRegression(penalty=penalty, C=c_reg, solver=solver, max_iter=max_iter)

            return model

        if self.model_name == 'RandomForest':
            # Return model initialization without setting hyperparams if optimization = True.
            if self._optimization:
                # Default hyperparameter ranges to explore in hyperparameter optimization.
                self.hyperparams_to_optimize = {'n_estimators': [50, 100, 200, 500, 1000],
                                                'max_depth':  [None, 5, 10, 20],
                                                'min_samples_split':  [2, 5, 10],
                                                'min_samples_leaf': [1, 2, 4],
                                                'max_features': ['auto', 'sqrt', 'log2', 0.5]}
                return RandomForestClassifier()

            # Hyperparameters to be used when defining the model.
            if kwargs is not None:
                n_estimators = kwargs['kwargs']['n_estimators']
                max_depth = kwargs['kwargs']['max_depth']
                min_samples_split = kwargs ['kwargs']['min_samples_split']
                min_samples_leaf = kwargs['kwargs']['min_samples_leaf']
                max_features = kwargs['kwargs']['max_features']
            else:
                n_estimators = 100 # Number of decision trees in Rforest.
                max_depth = None # Maximum depth of each decision tree.
                min_samples_split = 2 # Minimum number of samples to split an internal node.
                min_samples_leaf = 1 # Minimum number of samples to be at a leaf node.
                max_features = 'auto' # Number of features to consider when looking
                                      # for the best split.

            # Hyperparameters that will be used when defining the model.
            self.hyperparameters = {'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'max_features': max_features}

            # Model definition.
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           max_features=max_features
                                           )
            return model

        if self._model_name == 'DecisionTree':
            # Return model initialization without hyperparams if optimization = True.
            if self._optimization:
                # Default hyperparameter ranges to explore in hyperparameter optimization.
                self.hyperparams_to_optimize = {'criterion': ['gini', 'entropy'],
                                                'max_depth': [None, 5, 10, 20],
                                                'min_samples_split': [2, 5, 10],
                                                'min_samples_leaf': [1, 2, 4],
                                                'max_features': ['auto', 'sqrt', 'log2', 0.5]}
                return DecisionTreeClassifier()

            # Hyperparameters to be used when defining the model.
            criterion = kwargs.get('criterion', 'gini') # Function to measure quality of a
                                                              # split.
            max_depth = kwargs.get('max_depth', None) # Maximum depth of each decision tree.
            min_samples_split = kwargs.get('min_samples_split', 2) # Minimum number of samples
                                                                   # to split an internal node.
            min_samples_leaf = kwargs.get('min_samples_leaf', 1) # Minimum number of samples
                                                                 # to be at a leaf node.
            max_features = kwargs.get('max_features', 'auto') # Number of features to consider
                                                                  # when looking for the best split.

            # Hyperparameters that will be used when defining the model.
            self.hyperparameters = {'criterion': criterion,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'max_features': max_features}

            # Model definition.
            model = DecisionTreeClassifier(criterion=criterion,
                                           max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           max_features=max_features
                                           )

            return model

        if self._model_name == 'SVM':
            # Return model initialization without hyperparams if optimization = True.
            if self._optimization:
                self.hyperparams_to_optimize = {'C':[0.1, 1, 10],
                                                'kernel': ['linear', 'rbf', 'poly'],
                                                'gamma': ['scale', 'auto', 0.1, 1],
                                                'degree': [2, 3, 4],
                                                'class_weight': [None, 'balanced']}
                return SVC()

            # Hyperparameters to be used when defining the model.
            c_reg = kwargs.get('C', 1.0) # The regularization parameter.
            gamma = kwargs.get('gamma', 'scale') # Kernel coefficient.
            degree = kwargs.get('degree', 3) # The degree of the polynomial kernel
                                                   # function 'poly'.
            class_weight = kwargs.get('class_weight', None) # Weights associated with classes.
            kernel = kwargs.get('kernel', 'rbf') # Kernel function used to transform the input
                                                 # space into a higher-dimensional feature space.

            # Hyperparameters that will be used when defining the model.
            self.hyperparameters = {'C': c_reg,
                                    'kernel': kernel,
                                    'gamma': gamma,
                                    'degree': degree,
                                    'class_weight': class_weight}

            # Model definition.
            model = SVC(C=c_reg, kernel=kernel, gamma=gamma,
                        degree=degree, class_weight=class_weight)

            return model

        if self._model_name == 'NaiveBayes':
            # Hyperparameters to be used when defining the model.
            priors = kwargs.get('priors', None) # Can be used to provide prior probabilities
                                                # of the classes. Useful for imbalanced df.

            # Hyperparameters that will be used when defining the model.
            self.hyperparameters = {'priors': priors}

           # Model definition.
            model = GaussianNB(priors=priors) # Has no hyperparameters to optimize.

            return model

        return None
