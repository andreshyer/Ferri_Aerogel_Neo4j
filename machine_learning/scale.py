from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Scaler:
    @staticmethod
    def scale_data(x_test, x_train, x_val, scale):
        """
        Description: Scale train, test, split data according to the :param scale.
        :param x_test: test feature
        :param x_train: train feature
        :param x_val: val feature
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
            x_train = scale.fit_transform(x_train)
            x_test = scale.transform(x_test)
            if x_val is not None:
                x_val = scale.transform(x_val)
        return x_test, x_train, x_val
