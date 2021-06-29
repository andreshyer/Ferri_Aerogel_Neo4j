from pathlib import Path
from math import ceil, floor

from numpy import array, arange
from pandas import read_csv
from matplotlib import pyplot as plt


def plot_histogram(differences: array):
    bins = arange(floor(differences.min()), ceil(differences.max()), 1)

    plt.xticks([i for i in range(10)])
    plt.yticks([int(i*2) for i in range(11)])
    plt.hist(differences, bins=bins, edgecolor='black', linewidth=1.2)
    plt.title(f"Differences Between Rows from Same Paper (num of columns = {len(data.columns)})")
    plt.xlabel("Average number of differences")
    plt.ylabel("count")
    plt.savefig("Histogram_of_row_differences.png")


def count_dif_in_row(df, index_column):

    df = df.fillna("")  # Numpy NaN messes with set(), replace with empty string
    index_values = set(df[index_column].tolist())  # Gather all unique values in index column

    dif_list = []
    for index_value in index_values:

        # Gather all the rows that have a index value equal to that being searched
        indexed_rows = df.loc[df[index_column] == index_value]

        # The overall number of difference between all rows in the gathered rows
        num_of_differences = 0

        for column in df.columns:

            # All the unique values in the column of the gathered rows
            unique_values = set(indexed_rows[column].tolist())

            # There will always be one unique value in the set, so do not count the first on as a difference
            num_of_differences += (len(unique_values) - 1)

        # Take the average number of difference, since there can be different number of rows for each index value
        avg_num_of_differences = num_of_differences / len(indexed_rows)

        # Gather the difference values into a list to be put into a histogram
        dif_list.append(avg_num_of_differences)

    return array(dif_list)


if __name__ == "__main__":
    file_path = "files/si_aerogels/si_aerogel_AI_machine_readable_v2.csv"
    file_path = str(Path(__file__).parent / file_path)
    data = read_csv(file_path)

    index = "paper_id"
    hist_differences = count_dif_in_row(data, index)
    plot_histogram(hist_differences)
