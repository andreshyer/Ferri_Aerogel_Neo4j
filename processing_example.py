from pathlib import Path
from pandas import DataFrame, read_csv, concat

from machine_learning import Featurizer, ComplexDataProcessor, DataSplitter, Scaler, HyperTune, Grid, Regressor, train, graph, name


def cluster_data(data):

    def test_if_has(row):
        values = row.values
        for value in values:
            if value in si_precursor_subset:
                return True
        return False

    si_precursor_subset = ['TEOS']
    si_precursor_columns = ['Si Precursor (0)', 'Si Precursor (1)', 'Si Precursor (2)']
    data = data.loc[data[si_precursor_columns].apply(test_if_has, axis=1)]
    data = data.loc[data['Formation Method (0)'].isin(['Sol-gel'])]
    # data = data.loc[data['Surface Area (m2/g)'] < 1000]
    data.reset_index(drop=True, inplace=True)
    return data


def example_no_tune():
    """
    Example set up for running a non tuned model

    """
    dataset = r"si_aerogel_AI_machine_readable_v2.csv"
    folder = "si_aerogels"
    file_path = "files/si_aerogels/si_aerogel_AI_machine_readable_v2.csv/"

    data_path = str(Path(__file__).parent / file_path)
    data = read_csv(data_path)
    # y_columns = ['Surface Area (m2/g)', 'Thermal Conductivity (W/mK)']
    # drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
    #                 'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
    #                 'Average Pore Size (nm)']
    algorithm = 'rf'
    run_name = name.name(algorithm, dataset, folder, True, False)
    y_columns = ['Surface Area (m2/g)']
    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
            'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
                    'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)']
    paper_id_column = 'paper_id'

    #drop_columns.pop(len(drop_columns)-1)
    #paper_id_column = None

    data = cluster_data(data)
    data = data.drop([paper_id_column], axis=1)

    featurizer = Featurizer(df=data, y_columns=y_columns, columns_to_drop=drop_columns)
    data = featurizer.remove_xerogels()
    data = featurizer.remove_non_smiles_str_columns(suppress_warnings=True)  # TODO think of better way than dropping cols
    data = featurizer.replace_compounds_with_smiles()
    data = featurizer.featurize_molecules(method='rdkit2d')
    #data.to_csv("test_final_df.csv")
    #print(type(feature_list))
    data = featurizer.replace_nan_with_zeros()
    # data.to_csv('testing_data.csv')
    # complex_processor = ComplexDataProcessor(df=data, y_columns=y_columns)
    # feature_importances, important_columns = complex_processor.get_only_important_columns(number_of_models=5)
    # data = data[important_columns]

    splitter = DataSplitter(df=data, y_columns=y_columns,
                            train_percent=0.8, test_percent=0.2, val_percent=0, grouping_column=None,state=None)
    test_features, train_features, test_target, train_target, feature_list = splitter.split_data()  # Splitting data
    #print(len(train_features))
    test_features, train_features = Scaler().scale_data("std",train_features, test_features)   # Scaling features
    estimator = Regressor.get_regressor(algorithm)  # Get correct regressor (algorithm)
    
    predictions, predictions_stats, scaled_predictions, scaled_predictions_stats = train.train_reg(algorithm, estimator, train_features, train_target, test_features, test_target)  # Get predictions after training n times 
    
    #feature_list = list(data.columns)  # Feature list
    graph.pva_graph(predictions_stats, predictions, run_name)  # Get pva graph
    graph.impgraph_tree_algorithm(algorithm, estimator, feature_list, run_name)  # Get feature imporance based on algorithm


def example_tuned():
    """
    Example set up for running a tuned model

    """
    dataset = r"si_aerogel_AI_machine_readable_v2.csv"
    folder = "si_aerogels"
    file_path = "files/si_aerogels/si_aerogel_AI_machine_readable_v2.csv/"
    
    data_path = str(Path(__file__).parent / file_path)
    data = read_csv(data_path)
    
    algorithm = 'rf'
    run_name = name.name(algorithm, dataset, folder, True, False)  # Get target column
    
    y_columns = ['Surface Area (m2/g)', 'Thermal Conductivity (W/mK)']
    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
            'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
                    'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)']
    paper_id_column = 'paper_id'  # Group by papar option

    drop_columns.pop(len(drop_columns) - 1)
    paper_id_column = None

    data = cluster_data(data)
#    data = data.drop([paper_id_column], axis=1)

    featurizer = Featurizer(df=data, y_columns=y_columns, columns_to_drop=drop_columns)
    data = featurizer.remove_xerogels()
    data = featurizer.remove_non_smiles_str_columns(suppress_warnings=True)  # TODO think of better way than dropping cols
    data = featurizer.replace_compounds_with_smiles()
    data= featurizer.featurize_molecules(method='rdkit2d')
    #print(type(feature_list))
    data = featurizer.replace_nan_with_zeros()
    # data.to_csv('testing_data.csv')
    # complex_processor = ComplexDataProcessor(df=data, y_columns=y_columns)
    # feature_importances, important_columns = complex_processor.get_only_important_columns(number_of_models=5)
    # data = data[important_columns]

    splitter = DataSplitter(df=data, y_columns=y_columns,
                            train_percent=0.8, test_percent=0.2, val_percent=0, grouping_column=None,state=None)
    test_features, train_features, test_target, train_target, feature_list = splitter.split_data()  # Splitting into train and test
    #print(len(train_features))
    test_features, train_features = Scaler().scale_data("std",train_features, test_features)  # Scaling train and test test features
    grid = Grid.make_normal_grid(algorithm)  # Make grid for hyper tuning based on algorithm
    
    tuner = HyperTune(algorithm, train_features, train_target, grid, opt_iter=3, cv_folds=3)  # Get parameters for hyper tuning
    estimator, param, tune_score = tuner.hyper_tune(method="random")  # Hyper tuning the model
    
    predictions, predictions_stats, scaled_predictions, scaled_predictions_stats = train.train_reg(algorithm, estimator, train_features, train_target, test_features, test_target)  # Get prediction results from training the model n times
    
    #feature_list = list(data.columns)  # Get feature list
    graph.pva_graph(predictions_stats, predictions, run_name)  # Get pva graph
    graph.impgraph_tree_algorithm(algorithm, estimator, feature_list, run_name)  # Get feature importance graph based on algorithm



if __name__ == "__main__":
    example_no_tune()
    #example_tuned()
