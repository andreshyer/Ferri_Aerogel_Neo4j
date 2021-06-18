from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


class Scaler:
    @staticmethod
    def scale_data(scale, train_features, test_features, val_features=None):
        """
        Description: Scale train, test, split data according to the :param scale.
        :param test_features: test feature
        :param train_features: train feature
        :param val_features: val feature
        :param scale: scale method.
            std is for StandardScaler, minmax is for MinMaxScaler, None will skip
        :return:
        """
        if scale == "std":
            scale = StandardScaler()
        elif scale == "minmax":
            scale = MinMaxScaler()
        else:
            Exception("No option " + scale + " for data scaling")

        if scale is not None:
            train_features = scale.fit_transform(train_features)
            test_features = scale.transform(test_features)

#            test_features, train_features= np.array(test_features), np.array(train_features)
            if val_features is not None:
                val_features = scale.transform(val_features)
                
#                val_features = np.array(val_features)
                return test_features, train_features, val_features
        return test_features, train_features
