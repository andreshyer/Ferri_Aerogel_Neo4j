import numpy as np
from numpy.ma import MaskedArray
from sklearn import tree
import sklearn.utils.fixes
from skopt.space import Real, Integer, Categorical, Space


class Grid:
    """
    Grid object containing all hyper parameter grids for rf, gdb, keras nn and xgboost
    TODO: Add CNN, GAN
    """
    @staticmethod
    def rf_bayes_grid():
        """ Defines hyper parameters for random forest """

        # Define parameter grid for skopt BayesSearchCV
        bayes_grid = {
            'n_estimators': Integer(100, 2000),
            'max_features': Categorical(['auto', 'sqrt']),
            'max_depth': Integer(1, 30),
            'min_samples_split': Integer(2, 30),
            'min_samples_leaf': Integer(2, 30),
            'bootstrap': Categorical([True, False])
        }
        return bayes_grid

    @staticmethod
    def rf_normal_grid():
        # define variables to include in parameter grid for scikit-learn CV functions
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=20)]  # Number of trees
        max_features = ['auto', 'sqrt']  # Number of features to consider at every split
        max_depth = [int(x) for x in np.linspace(1, 30, num=11)]  # Maximum number of levels in tree
        min_samples_split = [2, 4, 6, 8, 10]  # Minimum number of samples required to split a node
        min_samples_leaf = [1, 2, 3, 4, 5, 6]  # Minimum number of samples required at each leaf node
        bootstrap = [True, False]  # Method of selecting samples for training each tree

        param_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }
        return param_grid

    @staticmethod
    def gdb_bayes_grid():
        """ Defines hyper parameters for gradient decent boost """

        # define variables to include in parameter grid for scikit-learn CV functions

        # Define parameter grid for skopt BayesSearchCV
        bayes_grid = {
            'n_estimators': Integer(500, 2000),
            'max_features': Categorical(['auto', 'sqrt']),
            'max_depth': Integer(1, 25),
            'min_samples_split': Integer(2, 30),
            'min_samples_leaf': Integer(2, 30),
            'learning_rate': Real(0.001, 1, 'log-uniform')
        }
        return bayes_grid

    @staticmethod
    def gdb_normal_grid():
        # Number of trees
        n_estimators = [int(x) for x in np.linspace(start=500, stop=2000, num=20)]

        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']

        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(1, 25, num=24, endpoint=True)]

        # Minimum number of samples required to split a node
        min_samples_split = [int(x) for x in np.linspace(2, 30, num=10, endpoint=True)]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [int(x) for x in np.linspace(2, 30, num=10, endpoint=True)]

        # learning rate
        learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

        param_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'learning_rate': learning_rate
        }
        return param_grid

    @staticmethod
    def keras_bayes_grid():
        bayes_grid = {
            'n_hidden': Integer(1, 10),
            'n_neuron': Integer(50, 300),
            'learning_rate': Real(0.0001, 0.1, 'log-uniform'),
            'drop': Real(0.1, 0.5)

        }
        return bayes_grid

    @staticmethod
    def keras_normal_grid():
        # Number of hidden nodes
        n_hidden = [int(x) for x in np.linspace(start=1, stop=30, num=30)]

        # Number of neurons
        n_neuron = [int(x) for x in np.linspace(start=50, stop=300, num=50)]

        # Learning rate
        learning_rate = [float(x) for x in np.linspace(start=0.0001, stop=0.1, num=50)]

        # Drop out rate
        drop = [float(x) for x in np.linspace(start=0.1, stop=0.5, num=10)]

        param_grid = {
            'n_hidden': n_hidden,
            "n_neuron": n_neuron,
            'learning_rate': learning_rate,
            'drop': drop
        }
        return param_grid

    @staticmethod
    def xgb_normal_grid():
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(9, 15, endpoint=True)]

        # Minimum weight required to create new node
        min_child_weight = [int(x) for x in np.linspace(5, 10, endpoint=True)]

        # Fraction of observations (the rows) to subsample at each step
        subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

        # Fraction of features (the columns) to use
        colsample_bytree = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

        # Learning rate
        eta = [.3, .2, .1, .05, .01, .005]

        param_grid = {
            'max_depth': max_depth,
            "min_child_weight": min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'eta': eta
        }
        return param_grid

    @staticmethod
    def xgb_bayes_grid():
        # Define parameter grid for skopt BayesSearchCV
        bayes_grid = {
            'max_depth': Integer(9, 15),
            'min_child_weight': Integer(9, 15),
            'subsample': Real(0.5, 1),
            'colsample_bytree': Real(0.5, 0.1),
            'eta': Real(0.001, 0.1, 'log-uniform')
        }
        return bayes_grid
