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

from machine_learning import Featurizer, ComplexDataProcessor, DataSplitter


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
    data = read_csv(str(Path(__file__).parent / "files/si_aerogels/si_aerogel_AI_machine_readable_v2.csv"))
    # y_columns = ['Surface Area (m2/g)', 'Thermal Conductivity (W/mK)']
    # drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
    #                 'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
    #                 'Average Pore Size (nm)']
    y_columns = ['Surface Area (m2/g)']
    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                    'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
                    'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)']
    paper_id_column = 'paper_id'


    # drop_columns.pop(len(drop_columns) - 1)
    # paper_id_column = None


    featurizer = Featurizer(df=data, y_columns=y_columns, columns_to_drop=drop_columns)
    data = featurizer.remove_xerogels()
    data = featurizer.remove_non_smiles_str_columns(suppress_warnings=True)  # TODO think of better way than dropping cols
    data = featurizer.replace_compounds_with_smiles()
    data = featurizer.featurize_molecules(method='rdkit2d')
    data = featurizer.replace_nan_with_zeros()
    data.to_csv('testing_data.csv')

    # data.to_csv('dev.csv')
    # featurizer = Featurizer(df=data, y_columns=y_columns, columns_to_drop=drop_columns)
    # featurizer.replace_words_with_numbers(ignore_smiles=True)
    # data = featurizer.replace_nan_with_zeros()

    # raise Exception('Stop')

    scalers = {'standard': StandardScaler(), 'min_max': MinMaxScaler()}
    models = {'rf': RandomForestRegressor(), 'nn': MLPRegressor()}
    featurized_string = 'featurized'

    for scaler_name, scaler in scalers.items():
        for model_name, model in models.items():

            subset_data = deepcopy(data)

            # complex_processor = ComplexDataProcessor(df=data, y_columns=y_columns)
            # feature_importances, important_columns = complex_processor.get_only_important_columns(number_of_models=5)
            # subset_data = subset_data[important_columns]

            splitter = DataSplitter(df=subset_data, y_columns=y_columns,
                                    train_percent=0.8, test_percent=0.2, val_percent=0,
                                    grouping_column=paper_id_column, state=None)
            x_test, x_train, x_val, y_test, y_train, y_val = splitter.split_data()
            # print(x_test)

            pva = DataFrame()
            pva['actual'] = y_test.values.tolist()

            x_scaler = deepcopy(scaler)
            x_scaler.fit(x_train)
            x_train = x_scaler.transform(x_train)
            x_test = x_scaler.transform(x_test)

            y_train = y_train.values.reshape(-1, 1)
            y_test = y_test.values.reshape(-1, 1)

            y_scaler = deepcopy(scaler)
            y_scaler.fit(y_train)
            y_train = y_scaler.transform(y_train).flatten()
            y_test = y_scaler.transform(y_test).flatten()

            predicted = DataFrame()
            for i in tqdm(range(5), desc="Predicting on data"):
                reg = deepcopy(model)
                reg.fit(x_train, y_train)
                y_predicted = reg.predict(x_test)
                y_predicted = y_scaler.inverse_transform(y_predicted.reshape(-1, 1))
                y_predicted = y_predicted.squeeze()
                predicted[f'predicted_{i}'] = y_predicted
            predicted = DataFrame(predicted)
            predicted_avg = predicted.mean(axis=1).tolist()
            predicted_std = predicted.std(axis=1).tolist()

            pva['pred_avg'] = predicted_avg
            pva['pred_std'] = predicted_std

            pva_graph(pva, f"{model_name}_{scaler_name}_{featurized_string}")

