from typing import Union

from pandas import DataFrame
from numpy import ndarray
from sklearn.model_selection import train_test_split
import numpy as np


class DataSplitter:

    def __init__(self, df: DataFrame, y_columns: Union[list[str], str],
                 train_percent: float, test_percent: float, val_percent: float = 0,
                 grouping_column: str = None, state: int = None):
        """
        The goal of this class is to take all the Data and split it up into a test, train, and val set

        :param df: Pandas DataFrame
        :param y_columns: The y column(s)
        :param train_percent: Percent of total dataset to be training
        :param test_percent: Percent to be testing
        :param val_percent: Percent to be val
        :param grouping_column: Column that group rows together (e.g. Title, paper id, id, etc.)
        :param state: int, Salt to reproduce the results
        """

        self.df: DataFrame = df
        self.train_percent: float = train_percent
        self.test_percent: float = test_percent
        self.val_percent: float = val_percent
        self.grouping_column: str = grouping_column
        self.state: int = state

        if not isinstance(y_columns, list):
            y_columns = [y_columns]
        self.y_columns: list[str] = y_columns

    @staticmethod
    def __reshape_df__(df: DataFrame):  # Test if dataframe only has one column, if so return a series with info
        """
        Check if the Pandas DataFrame has only one column, if it does cast it to a Pandas Series
        
        :param df: Pandas DataFrame
        :return: Pandas DataFrame, Pandas Series
        """
        if len(df.columns) == 1:
            return df.squeeze()
        else:
            return df

    def __do_split__(self, x, y, train_percent, test_percent, val_percent):
        """
        The function that actually splits up the data, and acts as a helper for __split_by_row__ and
        __split_by_group__

        :param x: x dataframe/series/list
        :param y: y dataframe/series/dummy_columns
        :param train_percent: percent for training
        :param test_percent: percent for testing
        :param val_percent: percent for val
        :return:
        """

        len_total = len(x)
        len_train = len_total * train_percent
        len_test = len_total * test_percent
        len_val = len_total * val_percent

        p1 = (len_train + len_val) / (len_train + len_test + len_val)  # Some /fancy/ math done here :)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=p1, random_state=self.state)

        if len_val > 0:
            p2 = len_train / (len_train + len_val)  # Here as well :P
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=p2,
                                                              random_state=self.state)
        else:
            x_val, y_val = None, None

        return x_test, x_train, x_val, y_test, y_train, y_val

    def __split_by_row__(self):
        """
        This function will split up the DataFrame by rows

        :return:
        """

        y = self.df[self.y_columns]
        x = self.df.drop(self.y_columns, axis=1)
        len_total = len(self.df)
        x_test, x_train, x_val, y_test, y_train, y_val = self.__do_split__(x, y,
                                                                           self.train_percent,
                                                                           self.test_percent,
                                                                           self.val_percent)
        return x_test, x_train, x_val, y_test, y_train, y_val

    def __split_by_group__(self):
        """
        This function will split up the DataFrame by groups (in our case, by papers)

        :return:
        """
        x_columns_to_drop = self.y_columns.copy()
        x_columns_to_drop.append(self.grouping_column)

        # Grab all the unique indexes in the grouping column (ex: paper_id)
        grouping_column_indexes = self.df[self.grouping_column].unique()

        # This behaves a lot like the code above, but we pass dummy lists for y, this is just for consistency with salt
        # https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
        dummy_y = range(len(grouping_column_indexes))
        test_ids, train_ids, val_ids, dummy_1, dummy_2, dummy_3 = self.__do_split__(grouping_column_indexes,
                                                                                    dummy_y,
                                                                                    self.train_percent,
                                                                                    self.test_percent,
                                                                                    self.val_percent)

        # cast numpy arrays to list
        test_ids, train_ids = list(test_ids), list(train_ids)

        # collect testing and training sets
        testing = self.df.loc[self.df[self.grouping_column].isin(test_ids)]
        training = self.df.loc[self.df[self.grouping_column].isin(train_ids)]

        # spilt up testing and training set into x and y
        x_test, y_test = testing.drop(x_columns_to_drop, axis=1), testing[self.y_columns]
        x_train, y_train = training.drop(x_columns_to_drop, axis=1), training[self.y_columns]

        # Do the same for val set if the val percent is more than 0
        if isinstance(val_ids, ndarray):  # If val_ids is not None
            val_ids = list(val_ids)
            val = self.df.loc[self.df[self.grouping_column].isin(val_ids)]
            x_val, y_val = val.drop(x_columns_to_drop, axis=1), val[self.y_columns]
        else:
            x_val, y_val = None, None
        return x_test, x_train, x_val, y_test, y_train, y_val

    def split_data(self):
        """
        This is the wrapper function that will class the above function and is the function that the user should
        call. This will split up the data by rows if a grouping column is not specified, and if a grouping
        column is specified, then the data will be split by unique keys in that column

        :return:
        """

        if self.grouping_column:
            x_test, x_train, x_val, y_test, y_train, y_val = self.__split_by_group__()
        else:
            x_test, x_train, x_val, y_test, y_train, y_val = self.__split_by_row__()

        # Verify the arrays are the correct shape
        x_test, y_test = self.__reshape_df__(x_test), self.__reshape_df__(y_test)
        x_train, y_train = self.__reshape_df__(x_train), self.__reshape_df__(y_train)
        if isinstance(x_val, DataFrame):  # If val sets are not
            x_val, y_val = self.__reshape_df__(x_val), self.__reshape_df__(y_val)

        x_test, x_train, x_val = np.array(x_test), np.array(x_train), np.array(x_val)
        
        if self.val_percent == 0.0:
            return x_test, x_train, y_test, y_train
        else:
            return  x_test, x_train, x_val, y_test, y_train, y_val
