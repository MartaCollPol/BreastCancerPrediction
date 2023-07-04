from sklearn.linear_model import LogisticRegression
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
        if not kwargs:
          self._hyperparameters = {}
        else:
          self._hyperparameters = kwargs['kwargs']
        self._default_hyperparams_to_optimize = None
        self._model_list = ['LogisticRegression',
                            'RandomForest', 'DecisionTree', 'Svm',
                            'NaiveBayes']

        assert self._model_name in self._model_list, (f"Model {model_name} is not available," +
                                                      f"possible models are {self._model_list}")

        # Apply class method based on model_name.
        method_name = 'define_' + ''.join(['_' + c.lower()
                                           if c.isupper() else c
                                           for c in model_name]).lstrip('_')
        self._model = getattr(self, method_name)()

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

    def define_logistic_regression(self):
        """
        Returns LogisticRegression model based on given or default hyperparameters. 
        """
        # Return model initialization without hyperparams if optimization = True.
        if self._optimization:
            # Default hyperparameter ranges to explore in hyperparameter optimization.
            self.hyperparams_to_optimize = {'penalty': ['l2'],
                                            'C': [1000, 10000, 20000, 30000],
                                            'solver': ['newton-cg',
                                                        'lbfgs',
                                                        'liblinear',
                                                        'sag',
                                                        'saga'],
                                            'max_iter': [5500, 6000, 7000]}                
            return LogisticRegression()

        # Hyperparameters to be used when defining the model.
        penalty = self._hyperparameters.get('penalty', 'l2') # Regularization penalty term.
        c_reg = self._hyperparameters.get('C', 1.0) # Inverse of regularization strength.
        solver = self._hyperparameters.get('solver', 'lbfgs') # Algorithm to use for optimization.
        max_iter = self._hyperparameters.get('max_iter', 100) # Maximum number of iterations
                                                              # for the solver to converge.
        self.hyperparameters = {'penalty': penalty,
                                'c_reg': c_reg,
                                'solver': solver,
                                'max_iter': max_iter}
        # Model definition.
        model = LogisticRegression(penalty=penalty, C=c_reg, solver=solver, max_iter=max_iter)

        return model

    def define_decision_tree(self):
        """
        Method to define a decision tree classifier based on default hyperparameters if not given.
        """
        # Return model initialization without hyperparams if optimization = True.
        if self._optimization:
            # Default hyperparameter ranges to explore in hyperparameter optimization.
            self.hyperparams_to_optimize = {'criterion': ['gini', 'entropy'],
                                            'max_depth': [None, 5, 10, 20],
                                            'min_samples_split': [2, 5, 10],
                                            'min_samples_leaf': [1, 2, 4],
                                            'max_features': ['sqrt', 'log2', 0.5]}
            return DecisionTreeClassifier()

        # Hyperparameters to be used when defining the model.
        criterion = self.hyperparameters.get('criterion', 'gini') # Function to measure quality of a
                                                                  # split.
        max_depth = self.hyperparameters.get('max_depth', None) # Maximum depth of each decision
                                                                # tree.
        min_samples_split = self.hyperparameters.get('min_samples_split', 2) # Minimum number of
                                                            # samples to split an internal node.
        min_samples_leaf = self.hyperparameters.get('min_samples_leaf', 1) # Minimum number of
                                                                # samples to be at a leaf node.
        max_features = self.hyperparameters.get('max_features', 'sqrt') # Number of features to
                                                    # consider when looking for the best split.

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
                                    max_features=max_features)

        return model

    def define_random_forest(self):
        """
        Method to define a Random Forest classifier based on default or given hyperparameters.
        """
        # Return model initialization without setting hyperparams if optimization = True.
        if self._optimization:
            # Default hyperparameter ranges to explore in hyperparameter optimization.
            self.hyperparams_to_optimize = {'n_estimators': [10, 20, 50, 100],
                                            'max_depth':  [None, 5, 10, 20],
                                            'min_samples_split':  [2, 5, 10],
                                            'min_samples_leaf': [1, 2, 4],
                                            'max_features': ['sqrt', 'log2', 0.5]}
            return RandomForestClassifier()

        # Hyperparameters to be used when defining the model.
        n_estimators = self.hyperparameters.get('n_estimators', 100)
        max_depth = self.hyperparameters.get('max_depth', None)
        min_samples_split = self.hyperparameters.get('min_samples_split', 2)
        min_samples_leaf = self.hyperparameters.get('min_samples_leaf', 1)
        max_features = self.hyperparameters.get('max_features', 'sqrt')

        # Hyperparameters that will be used when defining the model.
        self.hyperparameters = {'n_estimators': n_estimators, # Number of decision trees in Rforest.
                                'max_depth': max_depth, # Maximum depth of each decision tree.
                                'min_samples_split': min_samples_split, # Minimum number of samples
                                                                        # to split an internal node.
                                'min_samples_leaf': min_samples_leaf, # Minimum number of samples to
                                                                        # be at a leaf node.
                                'max_features': max_features} # Number of features to consider when
                                                                # looking for the best split.

        # Model definition.
        model = RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features
                                        )
        return model

    def define_svm(self):
        """
        Method to define SVM classifier based on given or default hyperparameteres. 
        """
        # Return model initialization without hyperparams if optimization = True.
        if self._optimization:
            self.hyperparams_to_optimize = {'C':[0.1, 1, 10],
                                            'kernel': ['linear', 'rbf', 'poly'],
                                            'gamma': ['scale', 'auto', 0.1, 1],
                                            'degree': [2, 3, 4],
                                            'class_weight': [None, 'balanced']}
            return SVC()

        # Hyperparameters to be used when defining the model.
        c_reg = self.hyperparameters.get('C', 1.0) # The regularization parameter.
        gamma = self.hyperparameters.get('gamma', 'scale') # Kernel coefficient.
        degree = self.hyperparameters.get('degree', 3) # The degree of the polynomial kernel
                                                # function 'poly'.
        class_weight = self.hyperparameters.get('class_weight', None) # Weights associated with classes.
        kernel = self.hyperparameters.get('kernel', 'rbf') # Kernel function used to transform the input
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

    def define_naive_bayes(self):
        """
        Method to define a Naive Bayes model based on given or default hyperparameters.
        """
        # Hyperparameters to be used when defining the model.
        priors = self.hyperparameters.get('priors', None) # Can be used to provide prior
                                # probabilities of the classes. Useful for imbalanced df.

        # Hyperparameters that will be used when defining the model.
        self.hyperparameters = {'priors': priors}

        # Model definition.
        model = GaussianNB(priors=priors) # Has no hyperparameters to optimize.

        return model
