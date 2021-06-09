from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split


class DataHolder:

    def __init__(self, df: pd.DataFrame, y: Union[list[str], str],
                 train_percent: float, test_percent: float, val_percent: float = None,
                 grouping_column: str = None, columns_to_drop: Union[list[str], str] = None,
                 state: int = None):
        # Gather data
        self.raw_df: pd.DataFrame = df
        self.df: pd.DataFrame = df
        self.grouping_column: str = grouping_column
        self.state: int = state

        # Make sure that y_columns are a list of strings
        if not isinstance(y, list):
            y = list(y)
        self.y_columns: list[str] = y

        # Make sure that all the y_columns specified exist in the DataFrame
        self.__check_data__()

        # Drop any columns that the user specifies
        if columns_to_drop:
            self.df = self.df.drop(columns_to_drop, axis=1)

        # Drop any rows that has a missing value in one on the y columns
        self.df = self.df.dropna(subset=self.y_columns, how='any')

        # Define train, test, val percent to 2 sig figs
        self.train_percent = round(train_percent, 2)
        self.test_percent = round(test_percent, 2)
        if val_percent:
            self.val_percent = round(val_percent, 2)
        else:
            self.val_percent = 0

        # Make sure the sum of percents add up to 1
        total_percent = self.test_percent + self.train_percent + self.val_percent
        if round(total_percent, 6) != 1:
            raise ValueError("The sum of test, train, and val percents do not add up to 1.")

    def __check_data__(self):
        bad_columns = []
        for y_column in self.y_columns:
            if y_column not in self.df.columns:
                bad_columns.append(y_column)
        if bad_columns:
            raise KeyError(f"{bad_columns} not found in axis")

    def replace_nan_with_zeros(self):
        self.df = self.df.fillna(0)  # Replace blank spaces with 0

    @ staticmethod
    def replace_words_with_numbers():
        df = pd.read_csv("dev.csv")
        df = df.dropna(axis=0, how='all')
        df.to_csv("dev.csv", index=False)

    def __do_split__(self, x, y, len_total, train_percent, test_percent, val_percent):

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

    def split_data(self):

        if not self.grouping_column:
            y = self.df[y_columns]
            x = self.df.drop(y_columns, axis=1)
            len_total = len(self.df)
            x_test, x_train, x_val, y_test, y_train, y_val = self.__do_split__(x, y, len_total, self.train_percent,
                                                                               self.test_percent, self.val_percent)

        else:

            # Grab all the unique indexes in the grouping column (ex: paper_id)
            grouping_column_indexes = self.df[self.grouping_column].unique()

            # This behaves a lot like the code above, but we pass dummy list for y, this is just for consistency
            # https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
            dummy_y = range(len(grouping_column_indexes))
            len_total = len(grouping_column_indexes)
            test_ids, train_ids, val_ids, dummy_1, dummy_2, dummy_3 = self.__do_split__(grouping_column_indexes,
                                                                                        dummy_y,
                                                                                        len_total,
                                                                                        self.train_percent,
                                                                                        self.test_percent,
                                                                                        self.val_percent)
            # cast numpy arrays to list
            test_ids, train_ids = list(test_ids), list(train_ids)

            # pandas apply function, create a column specifying weather the row is in the test, train, or val set
            def __gather_row_type__(i):
                if i in test_ids:
                    return 'test'
                elif i in train_ids:
                    return "train"
                elif i in val_ids:
                    return "val"
                else:
                    raise Exception("Something has broken badly here...")

            self.df['row_type'] = self.df[self.grouping_column].apply(__gather_row_type__)

            # Separate the rows into their respective sets
            testing = self.df.loc[self.df['row_type'] == "test"].drop(['row_type'], axis=1)
            training = self.df.loc[self.df['row_type'] == "train"].drop(['row_type'], axis=1)

            # Split the sets into testing and training data
            x_test, y_test = testing.drop(y_columns, axis=1), testing[y_columns]
            x_train, y_train = training.drop(y_columns, axis=1), training[y_columns]

            # Do the same for val set if the val percent is more than 0
            if val_ids:
                val_ids = list(val_ids)
                val = self.df.loc[self.df['row_type'] == "val"].drop(['row_type'], axis=1)
                x_val, y_val = val.drop(y_columns, axis=1), val[y_columns]
            else:
                x_val, y_val = None, None

            # Remove set identifying column from dataframe, can undo this if we want this info later
            self.df = self.df.drop(['row_type'], axis=1)

        return x_test, x_train, x_val, y_test, y_train, y_val


if __name__ == "__main__":
    data = pd.read_csv(str(Path(__file__).parent.parent / "files/si_aerogels/si_aerogel_AI_machine_readable.csv"))
    y_columns = ['Surface Area (m2/g)', 'Thermal Conductivity (W/mK)']
    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                    'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
                    'Average Pore Size (nm)']
    paper_id_column = "paper_id"
    holder = DataHolder(df=data, y=y_columns, columns_to_drop=drop_columns, grouping_column=paper_id_column,
                        train_percent=0.80, test_percent=0.20, val_percent=0)
    holder.replace_nan_with_zeros()
    holder.replace_words_with_numbers()
    # holder.split_data()
