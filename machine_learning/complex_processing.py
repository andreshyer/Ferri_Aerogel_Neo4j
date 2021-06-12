from typing import Union
from random import randint

from tqdm import tqdm
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor

from machine_learning import DataSplitter


class ComplexDataProcessor:

    def __init__(self, df: DataFrame, y_columns: Union[list[str], str], grouping_column: str = None,
                 state: int = None):
        """
        This class is a bit more to explain. So a random forest has an option to show what it think the importance
        of different columns in the training set are. So the overall goal of this class is to run a bunch of
        bad random forest on the data, and find columns that have end a small amount of importance for the machine.
        This is especially useful for Neural Networks since the data is so sparse.


        :param df: Pandas DataFrame
        :param y_columns: Column(s) to predict
        :param grouping_column: Column that group rows together (e.g. Title, paper id, id, etc.)
        :param state: int, Salt to reproduce the results
        """

        self.raw_df: DataFrame = df
        self.df: DataFrame = df
        self.grouping_column: str = grouping_column
        self.state = state

        if not isinstance(y_columns, list):
            y_columns = [y_columns]
        self.y_columns: list[str] = y_columns

        self.feature_importances: None = None

    @staticmethod
    def __parse_x_columns__(x_test: Union[DataFrame, Series]):
        """
        This function trys to parse the column names from the test set. This is actually to try and catch a bug that
        can come up if there is only one column in the x dataframe, as that would become a series. So, if the x sets
        have only one column (and hence is a pandas Series), just take name from Series and cast it to a list.

        :param x_test: Pandas DataFrame or Pandas Series
        :return: list of column names
        """
        if isinstance(x_test, DataFrame):
            x_columns = x_test.columns.tolist()
        elif isinstance(x_test, Series):
            x_columns = [x_test.name]
        else:
            raise Exception("Something has seriously gone wrong, the resulting x test set is "
                            "not a DataFrame or a Series.")
        return x_columns

    def __calculate_importances__(self, x_columns, number_of_models):
        """
        This function actually runs the random forest models and returns the average feature importance
        for each column in a DataFrame

        :param x_columns: The x_columns
        :param number_of_models: The number of random forest models to run
        :return: Pandas DataFrame
        """

        feature_importances = DataFrame()
        feature_importances['columns'] = x_columns
        feature_importances['importance'] = 0

        # Define the base state
        if not self.state:
            base_state = randint(0, 10000000)
        else:
            base_state = self.state

        # Gather the feature importances for each column
        for i in tqdm(range(number_of_models), desc="Finding important columns"):
            state = base_state + i
            splitter = DataSplitter(self.df, self.y_columns, train_percent=0.8, test_percent=0.2, state=state)
            x_test, x_train, x_val, y_test, y_train, y_val = splitter.split_data()
            reg = RandomForestRegressor()
            reg.fit(x_train, y_train)
            feature_importances['importance'] += reg.feature_importances_
        feature_importances['importance'] = feature_importances['importance'] / number_of_models
        return feature_importances

    def get_only_important_columns(self, number_of_models: int = 100, threshold: float = 0.01):
        """
        Kind of the wrapper class, the class that runs the functions above, then formats the initial DataFrame
        and saves the formatted feature importances DataFrame if the user wishes to fetch it.

        :param number_of_models:
        :param threshold:
        :return:
        """

        # parse x columns
        splitter = DataSplitter(self.df, self.y_columns, train_percent=0.8, test_percent=0.2, state=self.state)
        x_test, x_train, x_val, y_test, y_train, y_val = splitter.split_data()
        x_columns = self.__parse_x_columns__(x_test)

        # Calculate feature importances
        feature_importances = self.__calculate_importances__(x_columns, number_of_models)

        # Only grab rows that have a have a importance value above the threshold
        feature_importances = feature_importances.loc[feature_importances['importance'] > threshold]

        # Sort and clean up DataFrame
        feature_importances = feature_importances.sort_values(by="importance", ascending=False)
        feature_importances = feature_importances.reset_index(drop=True)

        # Store the feature importances if user wants to fetch results later
        self.feature_importances: DataFrame = feature_importances

        # Collect the important columns
        important_columns = feature_importances['columns'].tolist()

        # Add in the y_columns
        important_columns.extend(self.y_columns)

        # Add in the grouping column if it exist
        if self.grouping_column:
            important_columns.append(self.grouping_column)

        # Get only the important columns from the DataFrame
        return feature_importances, important_columns
