from pathlib import Path
from os import urandom

from numpy import nan
from pandas import read_csv

from machine_learning import Featurizer, DataSplitter, Scaler, name, Regressor, train, graph, Grid, HyperTune
from machine_learning.misc import zip_run_name_files


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

    featurizer = Featurizer(df=data, y_columns=y_columns, columns_to_drop=drop_columns)
    featurizer.remove_xerogels()

    if featurized_state:
        featurizer.remove_non_smiles_str_columns(suppress_warnings=True)
        featurizer.replace_compounds_with_smiles()
        data = featurizer.featurize_molecules(method='rdkit2d')
    else:
        data = featurizer.replace_words_with_numbers(ignore_smiles=False)

    if algorithm != "xgb":
        data = featurizer.replace_nan_with_zeros()

    splitter = DataSplitter(df=data, y_columns=y_columns, train_percent=train_percent,
                            test_percent=(1 - train_percent), val_percent=0, grouping_column=paper_id_column,
                            state=seed)
    test_features, train_features, test_target, train_target, feature_list = splitter.split_data()
    test_features, train_features = Scaler().scale_data("std", train_features, test_features)  # Scaling features

    if tuning_state:
        grid = Grid.make_normal_grid(algorithm)  # Make grid for hyper tuning based on algorithm
        tuner = HyperTune(algorithm, train_features, train_target, grid, opt_iter=50,
                          cv_folds=3)  # Get parameters for hyper tuning
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
    file_path = "files/si_aerogels/si_aerogel_AI_machine_readable_v2.csv/"

    data_path = str(Path(__file__).parent / file_path)
    data = read_csv(data_path)

    algorithms = ['nn', 'xgb', 'rf', 'gdb']
    tuning = [True, False]
    featurized = [True, False]
    clustered = [True, False]
    grouped = [True, False]

    train_percent = 0.8  # test_percent = 1 - train_percent

    # seed = int.from_bytes(urandom(3), "big")  # Generate an actual random number
    seed = None

    y_columns = ['Surface Area (m2/g)']
    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                    'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
                    'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)']
    paper_id_column = 'paper_id'  # Group by paper option

    main(data, seed)
