from typing import Union
from pathlib import Path
from warnings import warn

from numpy import isnan
from pandas import DataFrame, read_csv, Series
from sklearn.preprocessing import StandardScaler


class Ingester:

    def __init__(self, df: DataFrame, columns_to_drop: Union[list[str], str] = None):
        """
        The goal of this class is to take in the initial DataFrame, and so some initial data cleaning.
        Namely, replace the solvent names with SMILES, replace nan with zeros, and replace all the words with numbers.

        :param df:
        :param columns_to_drop:
        """
        self.raw_df: DataFrame = df
        self.df = df

        # Drop columns that are all nan
        self.df = self.df.dropna(axis=1, how='all')

        # Strip whitespace in DataFrame if there is any
        self.df: DataFrame = self.df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Drop columns specified by user
        if columns_to_drop:
            self.df: DataFrame = self.df.drop(columns_to_drop, axis=1)

        # Gather information in the compound info file
        compound_info_file = str(Path(__file__).parent.parent / "files/featurized_molecules/compound_info.csv")
        self.compound_df: DataFrame = read_csv(compound_info_file)
        self.compound_dict: dict[str, str] = dict(zip(self.compound_df['compound'], self.compound_df['smiles']))

        # Hold variable for sklearn-scaler
        self.scaler = None

    def __test_col_if_obj_smiles__(self, df_column: Series):
        """
        Test if a column is all smiles or not

        :param df_column: Pandas Series, from a DataFrame column most likely
        :return: bool, True if all SMILES
        """

        def test_value_if_smiles(x):
            if isinstance(x, int) or isinstance(x, float):
                if isnan(x):
                    return 'maybe'
            elif x in self.compound_dict.keys():
                return 'yes'
            elif x in self.compound_dict.values():
                return 'yes'
            return 'no'

        number_total = len(df_column)
        test_df_column = df_column.apply(test_value_if_smiles)
        number_yes = len(test_df_column[test_df_column == 'yes'])
        number_maybe = len(test_df_column[test_df_column == 'maybe'])  # Maybe is define as columns with 0

        if number_yes == 0:
            return False

        if number_yes != (number_total - number_maybe):
            print(df_column[test_df_column == 'no'].tolist())
            raise TypeError(f"There only some smiles found in column {df_column.name}")

        return True

    def replace_compounds_with_smiles(self, verify: bool = True):
        """
        This function parses each column in the DataFrame and searches for columns that are all compounds. Then
        it goes and replaces each of those compounds with their SMILES in compound_info.csv.

        :param verify: bool, option to make sure all the compounds in a column have a corresponding SMILES
        :return: Pandas DataFrame
        """

        def replace_name_with_smiles(x):
            if x in self.compound_dict.keys():
                return self.compound_dict[x]
            return x

        for column in self.df:
            if self.df[column].dtype == "object":
                if verify:
                    if self.__test_col_if_obj_smiles__(self.df[column]):
                        self.df[column] = self.df[column].apply(replace_name_with_smiles)
                else:
                    self.df[column] = self.df[column].apply(replace_name_with_smiles)

    def replace_nan_with_zeros(self):
        """
        Simple function, replace all blank cell with zeros.

        :return: Pandas DataFrame
        """
        self.df = self.df.fillna(0)  # Replace blank spaces with 0
        return self.df

    def replace_words_with_numbers(self, ignore_smiles: bool = True):
        """
        This function's goal is simple, replace every word with a corresponding number. Also, if specified,
        skip SMILES to be featurized later.

        :param ignore_smiles: bool, if True skip SMILES columns
        :return: Pandas DataFrame
        """

        keywords = [0]  # Force 0 to be the entry, as this will exist if replace_nan_with_zeros was run
        for column in self.df:
            if self.df[column].dtype == "object":  # If the column has any strings
                if not ignore_smiles:
                    if not self.__test_col_if_obj_smiles__(self.df[column]):  # If the column is not a SMILES column
                        keywords.extend(self.df[column].unique())  # Get all the unique values in the column
                else:
                    keywords.extend(self.df[column].unique())
        keywords = set(keywords)  # Keep on the unique words
        keywords = dict(zip(keywords, range(len(keywords))))  # Create a dict of [words, id]
        self.df = self.df.replace(keywords)  # Replace the words in the DataFrame
        return self.df

    def remove_non_smiles_str_columns(self, suppress_warnings=False):

        bad_columns = []
        for column in self.df:
            if self.df[column].dtype == "object":  # If the column has any strings
                if not self.__test_col_if_obj_smiles__(self.df[column]):
                    if not suppress_warnings:
                        warn(f"Dropping column {column}")
                    bad_columns.append(column)
        self.df = self.df.drop(bad_columns, axis=1)
        return self.df
