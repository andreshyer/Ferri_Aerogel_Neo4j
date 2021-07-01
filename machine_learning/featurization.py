from pathlib import Path
from typing import Union
from re import match
from tqdm import tqdm

from numpy import isnan, nan
from pandas import DataFrame, read_csv, concat

from machine_learning import Ingester


class Featurizer(Ingester):

    def __init__(self, df: DataFrame, y_columns: list, columns_to_drop: list = None):
        """
        The overall goal of this script is fundamentally different than a normal featurization pipeline.
        If we were to featurize each of the smiles separately, then that would result in thousands of columns,
        which is just not the best move.

        There are few things to consider.
            - Order matters for a machine
            - The number of columns has huge effects on the speed of machine learning

        If we look at the si_aerogel_AI_machine_readable.csv file, we see there are quite a few columns that are
        extensions of each other such as, Dopant (0), Dopant (1), Dopant (2), etc. Instead of trying to train on
        each featurized Dopant column, all Dopants are dopants, so the featurizations can be averaged across them. If
        the groups are averaged, then that only leaves a few hundred rows. This is still a hugh amount of columns, and
        the most important columns from the featurizations should be considered. But, this is a good start and addresses
        the issues from above.

        So the psuedo type of code for this class is

        Create a dict of {base_columns: [columns]}
        Where if the grouping columns are Dopant (0), Dopant (1), Dopant (2), then the dict would be
            {dopant: [Dopant (0), Dopant (1), Dopant (2)]}

        Then featurize each column in this dict, and create a new dict, so the new dict would be
            {dopant: [featurized Dopant (0), featurized Dopant (1), featurized Dopant (2)]}

        Then the average of each list of featurized groups is taken, so the new dict would be
            {dopant: average featurized Dopant} -> {dopant: af Dopant}

        There is more than just dopant as a group in the file, so it would actually look like
            {dopant: af Dopant, Additional Si Co-Precursor(s) : af Additional Si Co-Precursor(s) (2), ...}

        At this point, the dict is no longer need, the afs are just sent to their own list

        [af Dopant, af Additional Si Co-Precursor(s), af ... ]  # Dont create dict one step above

        Also, for certain columns that contain SMILES that are not part of a group, the average is just the featurized
            values, such as for Si Precursors

        [af Dopant, af Additional Si Co-Precursor(s), af Si Precursors, af ... ]

        Then the the afs are all concated together

        final af = concat_all(list of afs)

        Then the orginial df has the columns were featurized dropped, and the final af is concated to it. At this
        point, the data is ready to go. This class is an child of Ingest, so all the rest of the prep work is done
        there (remove_non_smiles_str_columns, replace_compounds_with_smiles).

        Then is recommend to run replace_nan_with_zeros after featurization, as that is the built in tool to that
        is ready right now. But, if in the future we wanted to replace that with other functions that
        scikit-learn has to offer, we can do that as well

        :param df: Pandas DataFrame
        """

        # Pull functions and class objects from Ingester
        super().__init__(df=df, y_columns=y_columns, columns_to_drop=columns_to_drop)
        self.columns_featurized: list[str] = []
        self.raw_df: DataFrame = self.df.copy()

    def reset_featurization(self):
        """
        This will remove all featurizations done, and will reset the featurizer object.
        Use with cation.

        :return: The original DataFrame inputted into featurizer object
        """

        self.df = self.raw_df.copy()
        return self.df

    def __gather_same_column_types__(self):
        """
        Gather the base columns and the columns that belong in that group

        If there is a SMILES column that stands alone, cast itself to it's own group

        :return: {dopant: [Dopant (0), Dopant (1), Dopant (2)...], ...}
        """

        grouped_columns = {}
        columns_featurized = []
        for column in self.df.columns:
            if self.__test_col_if_obj_smiles__(self.df[column]):  # Test if column is a SMILES column
                matcher = match(".*\(\d\)", column)  # Look for a number in parenthesis
                if matcher:  # If the column belongs to a group
                    base_column = column[:-4]  # Remove number in parenthesis
                    if base_column in grouped_columns.keys():  # If it already exist in the dict
                        grouped_columns[base_column].append(column)  # Create a new key entry with column
                    else:
                        grouped_columns[base_column] = [column]  # Else add the column to the group
                    columns_featurized.append(column)  # Keep track of all the columns that are being featurized on
                else:  # If not, cast it to it's own group grouped_columns[column] = [column]
                    columns_featurized.append(column)  # Keep track of all the columns that are being featurized on
        grouped_columns = grouped_columns.values()  # Only really need the values, the keys were just for organizing
        return grouped_columns, list(set(columns_featurized))

    def __featurize_each_df__(self, grouped_columns, pre_featurized_dict, pre_featurized_cols, nulls):
        """
        This function looks at all the columns from the function above, and featurizes the columns in
        self.df and puts them into a new dict.

        :param grouped_columns: {dopant: [Dopant (0), Dopant (1), Dopant (2)...], ...}
        :param pre_featurized_dict: {smiles: features}, from the featurized csv file
        :param pre_featurized_cols: [smiles, BalabanJ, BertzCT, ...]
        :param nulls: Array of zeros equal to length of featurized data
        :return: {dopant: [featurized Dopant (0), featurized Dopant (1), featurized Dopant (2)...], ...}
        """

        # Return either the featurized data if cell is not nan, or return array of zeros
        def featurize_singe_compound(x):
            if isinstance(x, float):
                if isnan(x):
                    return nulls
            return pre_featurized_dict[x]

        # Calculate the featurized dataframes
        featurized_dfs = []
        for list_of_cols in grouped_columns:
            grouped_featurized_dfs = []
            for column in list_of_cols:
                featurized_df = self.df[column].apply(featurize_singe_compound)  # Fetch feature for each compound
                featurized_df = featurized_df.tolist()  # Send Dense Series to a list of list
                featurized_df = DataFrame(featurized_df, columns=pre_featurized_cols)  # Turn list of list to a df
                featurized_df = featurized_df.drop(['smiles'], axis=1)  # Remove smiles column
                grouped_featurized_dfs.append(featurized_df)  # Collect featurized df in dict
            featurized_dfs.append(grouped_featurized_dfs)

        return featurized_dfs

    def __squeeze_grouped_dfs__(self, featurized_dfs):

        def calculate_mean(row):
            counter = 0
            total_value = 0
            for value in row.values:
                if not isnan(value):
                   total_value += value
                   counter += 1

            if counter == 0:
                return 0

            return total_value / counter

        squeezed_dfs = []
        for n, sub_featurized_dfs in enumerate(tqdm(featurized_dfs, desc="Featurizing Data")):

            squeezed_df = DataFrame()
            sub_featurized_df_columns = sub_featurized_dfs[0].columns
            for sub_featurized_column in sub_featurized_df_columns:

                df_with_only_column = DataFrame()
                for i, sub_featurized_df in enumerate(sub_featurized_dfs):
                    df_with_only_column[f"{i}"] = sub_featurized_df[sub_featurized_column]
                squeezed_df[sub_featurized_column] = df_with_only_column.apply(calculate_mean, axis=1)
            squeezed_df.columns = sub_featurized_df_columns + f" {self.columns_featurized[n]} [{n}]"

            squeezed_dfs.append(squeezed_df)
        return squeezed_dfs

    @staticmethod
    def __concat_squeezed_dfs__(squeezed_dfs):
        featurized_df = squeezed_dfs.pop(0)
        for squeezed_df in squeezed_dfs:
            featurized_df = concat((featurized_df, squeezed_df), axis=1)
        return featurized_df

    def featurize_molecules(self, method: str):
        """
        This is just the wrapper that calls the functions above to featurize the data. Please read the __init__
        doc string to get a feel for what this function is trying to do.

        :param method: descriptor name from descriptastorus
        :return: Featurized Pandas DataFrame
        """

        available_methods = ['atompaircounts', 'morgan3counts', 'morganchiral3counts', 'morganfeature3counts',
                             'rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits']

        if method not in available_methods:
            raise KeyError(f"User method of {method} is not a featuriziaton method."
                           f"The available methods are {available_methods}")

        # Fetch pre featurized data
        pre_featurized_data_file = str(Path(__file__).parent.parent / f"files/featurized_molecules/{method}.csv")
        pre_featurized_data = read_csv(pre_featurized_data_file)
        pre_featurized_data = pre_featurized_data.fillna(0)

        # Holding this code here if we want to only featurize with rdkit2d and only use certain columns
        # only_consider_columns = []
        # if only_consider_columns:
        #     pre_featurized_data = pre_featurized_data[only_consider_columns]

        # Create a list of nan's the length of the featurized
        nulls = [nan] * len(pre_featurized_data.columns)

        # Create the {smiles: features} object
        pre_featurized_dict = dict(zip(pre_featurized_data['smiles'].values.tolist(),
                                       pre_featurized_data.values.tolist()))

        # Fetch columns in the pre featurized data
        pre_featurized_cols = pre_featurized_data.columns

        # Run the functions to featurize the data
        grouped_columns, self.columns_featurized = self.__gather_same_column_types__()
        featurized_df = self.__featurize_each_df__(grouped_columns, pre_featurized_dict, pre_featurized_cols,
                                                   nulls)
        featurized_df = self.__squeeze_grouped_dfs__(featurized_df)
        featurized_df = self.__concat_squeezed_dfs__(featurized_df)

        self.df = self.df.drop(self.columns_featurized, axis=1)  # Remove the columns that we featurized

        self.df.reset_index(drop=True, inplace=True)  # Reset the indexes for the dataframes
        featurized_df.reset_index(drop=True, inplace=True)
        self.df = concat((self.df, featurized_df), axis=1)  # Concat the main df to the featurized df
        return self.df


def featurize_si_aerogels(df: DataFrame, str_method: str, num_method: str, y_columns: list, drop_columns: list = None,
                          remove_xerogels: bool = True):

    str_methods = ["rdkit", "one_hot_encode", "number_index"]
    num_methods = ["zeros", "smart_values", "mean"]

    if str_method not in str_methods:
        raise TypeError(f"str_method {str_method} to replace strings with numbers not found in string methods."
                        f"\nList of available string methods are {str_methods}")

    if num_method not in num_methods:
        raise TypeError(f"num_method {num_method} to replace nan with in number columns not found in number methods."
                        f"\nList of available number methods are {num_methods}")

    # Define featurizer class to do calculations with
    featurizer = Featurizer(df=df, y_columns=y_columns, columns_to_drop=drop_columns)
    df = featurizer.df

    if remove_xerogels:
        df = featurizer.remove_xerogels()

    # Featurize the string columns
    if str_method == "one_hot_encode":
        df = featurizer.one_hot_encode_strings()
    if str_method == "rdkit":
        featurizer.remove_non_smiles_str_columns(suppress_warnings=True)  # TODO think of better way to do this
        featurizer.replace_compounds_with_smiles()
        df = featurizer.featurize_molecules(method='rdkit2d')
    if str_method == "number_index":
        df = featurizer.replace_words_with_numbers(ignore_smiles=False)

    # Featurize the number columns
    if num_method == "zeros":
        df = featurizer.replace_nan_with_zeros()
    if num_method == "mean":
        df = featurizer.replace_cols_with_nan_with_mean(cols=df.columns)
    if num_method == "smart_values":

        # Set NaN in temp columns as room temperature
        temp_columns = list(df.filter(regex="Temp").columns)
        featurizer.replace_cols_with_nan_with_number(cols=temp_columns, num=25)

        # Set NaN in pressure columns as atmospheric pressure
        pressure_columns = list(df.filter(regex="Pressure").columns)
        featurizer.replace_cols_with_nan_with_number(cols=pressure_columns, num=0.101325)

        # Set columns to zero where averages do not make sense
        ratio_columns = list(df.filter(regex="Ratio").columns)
        featurizer.replace_cols_with_nan_with_number(cols=ratio_columns, num=0)

        ratio_columns = list(df.filter(regex="%").columns)
        featurizer.replace_cols_with_nan_with_number(cols=ratio_columns, num=0)

        ph_columns = list(df.filter(regex="pH").columns)
        featurizer.replace_cols_with_nan_with_number(cols=ph_columns, num=7)

        # Set columns to averages where it makes more sense than zero or a set value
        time_columns = list(df.filter(regex="Time").columns)
        featurizer.replace_cols_with_nan_with_mean(cols=time_columns)

        time_columns = list(df.filter(regex="time").columns)  #
        featurizer.replace_cols_with_nan_with_mean(cols=time_columns)

        molar_columns = list(df.filter(regex="\(M\)").columns)
        featurizer.replace_cols_with_nan_with_mean(cols=molar_columns)

        rate_columns = list(df.filter(regex="Rate").columns)  #
        featurizer.replace_cols_with_nan_with_mean(cols=rate_columns)

        duration_columns = list(df.filter(regex="Duration").columns)
        df = featurizer.replace_cols_with_nan_with_mean(cols=duration_columns)

    return df
