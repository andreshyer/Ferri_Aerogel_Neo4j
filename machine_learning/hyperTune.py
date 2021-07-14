from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from machine_learning.regressor import Regressor
import skopt
from skopt import BayesSearchCV
from skopt import callbacks

from machine_learning.keras_nn import HyperTuneKeras


class HyperTune:
    """
    TODO: Make sure Bayes work
    """

    def __init__(self, algorithm, train_features, train_target, param_grid, opt_iter=10, n_jobs=3,
                 cv_folds=3, scoring="neg_mean_squared_error", deltay=None, fit_params=None):

        self.algorithm = algorithm
        self.estimator = Regressor.get_regressor(self.algorithm)
        self.train_features = train_features
        self.train_target = train_target
        self.param_grid = param_grid
        self.opt_iter = opt_iter
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.deltay = deltay
        self.fit_params = fit_params

    def hyper_tune(self, method="random"):
        """
        """

        if self.algorithm == "nn":
            tuner = HyperTuneKeras(self.train_features, self.train_target,
                                   seed=None, validation_percent=0.1)
            estimator, params = tuner.tune()
            return tuner, estimator, params

        elif method == "random":
            tune_algorithm = RandomizedSearchCV(estimator=self.estimator, param_distributions=self.param_grid,
                                                n_iter=self.opt_iter, scoring=self.scoring, random_state=42,
                                                n_jobs=self.n_jobs, cv=self.cv_folds)
            tune_algorithm.fit(self.train_features, self.train_target)
        elif method == "bayes":
            tune_algorithm = BayesSearchCV(estimator=self.estimator,  # what regressor to use
                                           search_spaces=self.param_grid,  # hyper parameters to search through
                                           fit_params=self.fit_params,
                                           n_iter=self.opt_iter,  # number of combos tried
                                           random_state=42,  # random seed
                                           scoring=self.scoring,  # scoring function to use (RMSE)
                                           n_jobs=self.n_jobs,  # number of parallel jobs (max = folds)
                                           cv=self.cv_folds
                                           # number of cross-val folds to use
                                           )

            cp_delta = float((0.05 - min(self.train_target.min())) / (
                    max(self.train_target.max()) - min(self.train_target.min())))  # Min max scaling

            n_best = 5
            callback = callbacks.DeltaYStopper(cp_delta, n_best)
            tune_algorithm.fit(self.train_features, self.train_target, callback=callback)

        else:
            raise TypeError("No tuning method called " + method)

        params = tune_algorithm.best_params_
        tune_score = tune_algorithm.best_score_

        estimator = Regressor.get_regressor(self.algorithm, given_param=params)

        return estimator, params, tune_score
