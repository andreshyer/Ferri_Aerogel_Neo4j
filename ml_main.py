from pathlib import Path
from os import urandom

from numpy import nan, isnan, array
from pandas import read_csv, read_excel

from machine_learning import Featurizer, DataSplitter, Scaler, name, Regressor, train, graph, Grid, HyperTune
from machine_learning.misc import zip_run_name_files
from machine_learning.featurization import featurize_si_aerogels
from backends.molarity_calculator import convert_machine_readable


def cluster_data(data):
    def test_if_has(row):
        values = row.values
        for value in values:
            if value in si_precursor_subset:
                return True
        return False

    def reject_outliers(value):
        if abs(value - mean_surface_area) < (3 * std_surface_area):
            return True
        return False

    si_precursor_subset = ['TEOS']
    si_precursor_columns = ['Si Precursor (0)', 'Si Precursor (1)', 'Si Precursor (2)']
    data = data.loc[data[si_precursor_columns].apply(test_if_has, axis=1)]  # Grab Aerogels with TEOS
    data = data.loc[data['Formation Method (0)'].isin(['Sol-gel'])]  # Grab sol-gel aerogels
    data = data.loc[data['Formation Method (1)'].isin([nan])]  # Make sure that is the only formation method

    # Remove outliers
    mean_surface_area = data['Surface Area (m2/g)'].mean()
    std_surface_area = data['Surface Area (m2/g)'].std()
    data = data.loc[data['Surface Area (m2/g)'].apply(reject_outliers)]
    # data = data.loc[data['Surface Area (m2/g)'] < 1000]

    data.reset_index(drop=True, inplace=True)
    return data


def run_params(data, run_name, y_columns, drop_columns, paper_id_column, train_percent, algorithm,
               tuning_state, featurized_state, clustered_state, grouped_state, seed):
    if clustered_state:
        data = cluster_data(data)

    if not grouped_state:
        data = data.drop([paper_id_column], axis=1)
        paper_id_column = None

    print("Featurizing Data...")
    if featurized_state:
        data = featurize_si_aerogels(df=data, str_method="rdkit", num_method="mean",
                                     y_columns=y_columns, drop_columns=drop_columns, remove_xerogels=True)

    else:
        data = featurize_si_aerogels(df=data, str_method="one_hot_encode", num_method="smart_values",
                                     y_columns=y_columns, drop_columns=drop_columns, remove_xerogels=True)

    splitter = DataSplitter(df=data, y_columns=y_columns, train_percent=train_percent,
                            test_percent=(1 - train_percent), val_percent=0, grouping_column=paper_id_column,
                            state=seed)
    test_features, train_features, test_target, train_target, feature_list = splitter.split_data()
    test_features, train_features = Scaler().scale_data("std", train_features, test_features)  # Scaling features

    print("Tuning Model...")
    if tuning_state:
        grid = Grid.make_normal_grid(algorithm)  # Make grid for hyper tuning based on algorithm
        tuner = HyperTune(algorithm, train_features, train_target, grid, opt_iter=50,
                          cv_folds=5)  # Get parameters for hyper tuning
        estimator, params, tune_score = tuner.hyper_tune(method="random")  # Hyper tuning the model
    else:
        estimator = Regressor.get_regressor(algorithm)  # Get correct regressor (algorithm)
        params = None
        tune_score = None

    # Get predictions after training n times
    predictions, predictions_stats, scaled_predictions, scaled_predictions_stats = train.train_reg(algorithm,
                                                                                                   estimator,
                                                                                                   train_features,
                                                                                                   train_target,
                                                                                                   test_features,
                                                                                                   test_target,
                                                                                                   fit_params=params,
                                                                                                   n=5)
    graph.pva_graph(predictions_stats, predictions, run_name)  # Get pva graph
    graph.pva_graph(scaled_predictions_stats, scaled_predictions, run_name, scaled=True)
    graph.shap_impgraphs(algorithm, estimator, train_features, feature_list, run_name)
    zip_run_name_files(run_name)


def main(data, seed):
    for clustered_state in clustered:
        for grouped_state in grouped:
            for featurized_state in featurized:
                for algorithm in algorithms:
                    for tuning_state in tuning:
                        run_name = name(algorithm, dataset, folder, featurized, tuning)
                        run_name += f"{grouped_state}"
                        run_params(data, run_name, y_columns, drop_columns, paper_id_column, train_percent,
                                   algorithm, tuning_state, featurized_state, clustered_state,
                                   grouped_state, seed)


if __name__ == "__main__":
    dataset = r"si_aerogel_AI_machine_readable_v2.csv"
    folder = "si_aerogels"
    file_path = "files/si_aerogels/si_aerogels.xlsx/"

    data_path = str(Path(__file__).parent / file_path)

    print("Gathering data...")
    data = convert_machine_readable(data_path)

    algorithms = ['rf']
    tuning = [False]
    featurized = [False]
    clustered = [False]
    grouped = [False]

    train_percent = 0.8  # test_percent = 1 - train_percent

    # seed = int.from_bytes(urandom(3), "big")  # Generate an actual random number
    seed = None

    y_columns = ['Gelation Time (mins)']

    author_columns = list(data.filter(regex="Author").columns)
    notes_columns = list(data.filter(regex="Notes").columns)
    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                    'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase',
                    'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)', 'Surface Area (m2/g)',
                    'Final Material', "Year", "Cited References (#)", "Times Cited (#)",
                    "pH final sol"]
    drop_columns.extend(author_columns)
    drop_columns.extend(notes_columns)

    paper_id_column = 'Title'  # Group by paper option

    main(data, seed)
