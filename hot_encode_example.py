from pathlib import Path
from os import urandom

from numpy import nan, isnan
from pandas import read_csv

from machine_learning import Featurizer, DataSplitter, Scaler, name, Regressor, train, graph, Grid, HyperTune
from machine_learning.misc import zip_run_name_files


def main():
    file_path = "files/si_aerogels/si_aerogel_AI_machine_readable_v2.csv/"

    data_path = str(Path(__file__).parent / file_path)
    data = read_csv(data_path)

    y_columns = ['Surface Area (m2/g)']
    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                    'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
                    'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)', 'Gelation Time (mins)', 'paper_id']
    paper_id_column = None

    train_percent = 0.8  # test_percent = 1 - train_percent
    seed = None
    algorithm = 'rf'
    run_name = "encode_example"

    print("Featurizing Data...")

    featurizer = Featurizer(df=data, y_columns=y_columns, columns_to_drop=drop_columns)

    # Drop xerogels, grab DataFrame here to make sure columns actually match up.
    data = featurizer.remove_xerogels()

    # Set NaN in temp columns as room temperature
    temp_columns = list(data.filter(regex="Temp").columns)
    featurizer.replace_cols_with_nan_with_number(cols=temp_columns, num=20)

    # Set NaN in pressure columns as atmospheric pressure
    pressure_columns = list(data.filter(regex="Pressure").columns)
    featurizer.replace_cols_with_nan_with_number(cols=pressure_columns, num=0.101325)

    # Set columns to zero where averages do not make sense
    ratio_columns = list(data.filter(regex="Ratio").columns)
    featurizer.replace_cols_with_nan_with_number(cols=ratio_columns, num=0)

    ratio_columns = list(data.filter(regex="%").columns)
    featurizer.replace_cols_with_nan_with_number(cols=ratio_columns, num=0)

    ph_columns = list(data.filter(regex="pH").columns)
    featurizer.replace_cols_with_nan_with_number(cols=ph_columns, num=7)

    # Set columns to averages where it makes more sense than zero or a set value
    time_columns = list(data.filter(regex="Time").columns)
    featurizer.replace_cols_with_nan_with_mean(cols=time_columns)

    time_columns = list(data.filter(regex="time").columns)
    featurizer.replace_cols_with_nan_with_mean(cols=time_columns)

    molar_columns = list(data.filter(regex="\(M\)").columns)
    featurizer.replace_cols_with_nan_with_mean(cols=molar_columns)

    rate_columns = list(data.filter(regex="rate").columns)
    featurizer.replace_cols_with_nan_with_mean(cols=rate_columns)

    duration_columns = list(data.filter(regex="Duration").columns)
    featurizer.replace_cols_with_nan_with_mean(cols=duration_columns)

    # Hot encode all string columns, grab DataFrame.
    # Data is fully featurized at this point
    data = featurizer.one_hot_encode_strings()

    splitter = DataSplitter(df=data, y_columns=y_columns, train_percent=train_percent,
                            test_percent=(1 - train_percent), val_percent=0, grouping_column=paper_id_column,
                            state=seed)
    test_features, train_features, test_target, train_target, feature_list = splitter.split_data()
    test_features, train_features = Scaler().scale_data("std", train_features, test_features)  # Scaling features

    print("HyperTuning Model...")

    grid = Grid.make_normal_grid(algorithm)  # Make grid for hyper tuning based on algorithm
    tuner = HyperTune(algorithm, train_features, train_target, grid, opt_iter=5,
                      cv_folds=3)  # Get parameters for hyper tuning
    estimator, params, tune_score = tuner.hyper_tune(method="random")  # Hyper tuning the model

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


if __name__ == "__main__":
    main()
