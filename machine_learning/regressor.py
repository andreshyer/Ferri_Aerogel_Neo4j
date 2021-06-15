from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class Regressor:
    """
    Regressor Object that returns the appropriate regressor based on name and given parameters
    TODO: Add val_features and val_target for Neural Network
    """
    @staticmethod
    def get_regressor(algorithm, call=None, given_param=None, val_features=None, val_target=None):

        """
        Returns model specific regressor function.
        Optional argument to create callable or instantiated instance.
        :param algorithm: ALgorithm to use
        :param given_param: Given hyper parameter
        :param val_features: Validation Feature
        :param val_target: validation target
        """
        
        skl_regs ={'rf': RandomForestRegressor,
                   'gdb': GradientBoostingRegressor,
                   'xgb': XGBRegressor
                   }                    
        if algorithm in skl_regs.keys():
            print(algorithm)
            if call:
                estimator = skl_regs[algorithm]
            else:
                if given_param is not None:  # has been tuned
                    estimator = skl_regs[algorithm](**given_param)

                else:
                    estimator = skl_regs[algorithm]()
            return estimator
