from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

import numpy as np
from pandas import DataFrame, read_csv, concat
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import cm
from machine_learning import Featurizer, ComplexDataProcessor, DataSplitter, Scaler, HyperTune, Grid, Regressor, train


def pva_graph(pva, run_name):
    """
    Make Predicted vs. Actual graph with prediction uncertainty.
    Pass dataframe from multipredict function. Return a graph.
    """

    r2 = r2_score(pva['actual'], pva['pred_avg'])
    mse = mean_squared_error(pva['actual'], pva['pred_avg'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))

    plt.rcParams['figure.figsize'] = [12, 9]
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    norm = cm.colors.Normalize(vmax=pva['pred_std'].max(), vmin=pva['pred_std'].min())
    x = pva['actual']
    y = pva['pred_avg']
    plt.scatter(x, y, c=pva['pred_std'], cmap='plasma', norm=norm, alpha=0.7)
    cbar = plt.colorbar()
    cbar.set_label("Uncertainty")

    # set axis limits
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
            ]

    # ax = plt.axes()
    plt.xlabel('True', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.title(run_name + f' Predicted vs. Actual')

    plt.plot(lims, lims, 'k-', label='y=x')
    plt.plot([], [], ' ', label='R^2 = %.3f' % r2)
    plt.plot([], [], ' ', label='RMSE = %.3f' % rmse)
    plt.plot([], [], ' ', label='MSE = %.3f' % mse)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # plt.axis([-2,5,-2,5]) #[-2,5,-2,5]
    ax.legend(prop={'size': 16}, facecolor='w', edgecolor='k', shadow=True)

    fig.patch.set_facecolor('blue')  # Will change background color
    fig.patch.set_alpha(0.0)  # Makes background transparent

    plt.savefig(run_name + '_' + f'PVA.png')
    plt.close()
    # plt.show()
    # self.pva_graph = plt
    # return plt


if __name__ == "__main__":
    data = read_csv(str(Path(__file__).parent / "files/si_aerogels/si_aerogel_AI_machine_readable.csv"))
    # y_columns = ['Surface Area (m2/g)', 'Thermal Conductivity (W/mK)']
    # drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
    #                 'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
    #                 'Average Pore Size (nm)']
    y_columns = ['Surface Area (m2/g)']
    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
            'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
                    'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)']
    paper_id_column = 'paper_id'


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
    data = data.loc[data['Surface Area (m2/g)'] < 1000]
    data.reset_index(drop=True, inplace=True)

    drop_columns.pop(len(drop_columns) - 1)

    featurizer = Featurizer(df=data, y_columns=y_columns, columns_to_drop=drop_columns)
    data = featurizer.remove_xerogels()
    data = featurizer.remove_non_smiles_str_columns(suppress_warnings=True)  # TODO think of better way than dropping cols
    data = featurizer.replace_compounds_with_smiles()
    data = featurizer.featurize_molecules(method='rdkit2d')

    data = featurizer.replace_nan_with_zeros()
    # data.to_csv('testing_data.csv')

    # complex_processor = ComplexDataProcessor(df=data, y_columns=y_columns)
    # feature_importances, important_columns = complex_processor.get_only_important_columns(number_of_models=5)
    # data = data[important_columns]

    splitter = DataSplitter(df=data, y_columns=y_columns,
                            train_percent=0.8, test_percent=0.2, val_percent=0, grouping_column=None,state=None)
    test_features, train_features, test_target, train_target = splitter.split_data()
    print(len(train_features))    
    #x_test, x_train = Scaler().scale_data("std",x_test, x_train)
    #print(len(x_test), len(x_train), len(y_test), len(y_train))

    #grid = Grid.rf_bayes_grid()
    #tuner = HyperTune("rf", x_train, y_train, grid, opt_iter=50)
    #estimator, param, tune_score = tuner.hyper_tune(method="random")
    estimator = Regressor.get_regressor("xgb")
    
    predictions, predictions_stats, scaled_predictions, scaled_predictions_stats = train.train_reg("xgb", estimator, train_features, train_target, test_features, test_target) 

    # pva = DataFrame()
    # pva['actual'] = y_test.values.tolist()
    #
    # x_scaler = StandardScaler()
    # x_scaler.fit(x_train)
    # x_train = x_scaler.transform(x_train)
    # x_test = x_scaler.transform(x_test)
    #
    # y_train = y_train.values.reshape(-1, 1)
    # y_test = y_test.values.reshape(-1, 1)
    #
    # y_scaler = StandardScaler()
    # y_scaler.fit(y_train)
    # y_train = y_scaler.transform(y_train).flatten()
    # y_test = y_scaler.transform(y_test).flatten()
    #
    # predicted = DataFrame()
    # for i in tqdm(range(100), desc="Predicting on data"):
    #     reg = MLPRegressor()
    #     reg.fit(x_train, y_train)
    #     y_predicted = reg.predict(x_test)
    #     y_predicted = y_scaler.inverse_transform(y_predicted)
    #     predicted[f'predicted_{i}'] = y_predicted
    # predicted = DataFrame(predicted)
    # predicted_avg = predicted.mean(axis=1).tolist()
    # predicted_std = predicted.std(axis=1).tolist()
    #
    # pva['pred_avg'] = predicted_avg
    # pva['pred_std'] = predicted_std
    #
    # pva_graph(pva, "split_by_group_nn")
