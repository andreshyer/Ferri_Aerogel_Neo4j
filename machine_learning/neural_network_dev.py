from pathlib import Path
from random import randint
from shutil import rmtree

from numpy import nan

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
from keras_tuner import Hyperband, RandomSearch
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

from machine_learning import Featurizer, DataSplitter, Scaler, graph


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


def single_out(data):
    new_df = DataFrame(columns=data.columns)

    paper_id_columns = set(data[paper_id_column].tolist())
    for col in paper_id_columns:
        rows = data.loc[data[paper_id_column] == col].to_dict('records')
        row = rows[randint(0, len(rows) - 1)]
        new_df = new_df.append(row, ignore_index=True)
    return new_df


# gather data
dataset = r"si_aerogel_AI_machine_readable_v2.csv"
folder = "si_aerogels"
file_path = "files/si_aerogels/si_aerogel_AI_machine_readable_v2.csv/"

data_path = str(Path(__file__).parent.parent / file_path)
data = read_csv(data_path)

y_columns = ['Surface Area (m2/g)']
drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
                'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)']
paper_id_column = 'paper_id'  # Group by paper option

data = single_out(data)
# data = cluster_data(data)

data = data.drop([paper_id_column], axis=1)
paper_id_column = None

featurizer = Featurizer(df=data, y_columns=y_columns, columns_to_drop=drop_columns)
featurizer.remove_xerogels()
# featurizer.remove_non_smiles_str_columns(suppress_warnings=True)
# featurizer.replace_compounds_with_smiles()
# data = featurizer.featurize_molecules(method='rdkit2d')

# featurizer.drop_all_word_columns(suppress_warning=True)
# data = featurizer.replace_nan_with_zeros()

featurizer.replace_words_with_numbers(ignore_smiles=False)
data = featurizer.replace_nan_with_zeros()

splitter = DataSplitter(df=data, y_columns=y_columns, train_percent=70,
                        test_percent=20, val_percent=10, grouping_column=paper_id_column,
                        state=None)
x_test, x_train, x_val, y_test, y_train, y_val, feature_list = splitter.split_data()

x_test, x_train, x_val = Scaler().scale_data("std", x_train, x_test, x_val)  # Scaling features
y_test, y_train, y_val = Scaler().scale_data("std", y_train, y_test, y_val)  # Scaling features


# model = Sequential()
# model.add(Dense(4, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1, name='output_layer'))
# model.compile(loss='mse', optimizer='adam', metrics=["mse", "mae"])
#
# history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val))
#
# plt.plot(history.history['mae'], '-*')
# plt.plot(history.history['val_mae'], '--o')
# plt.title('MAE: Train and Validation', fontsize=14, fontweight='bold')
# plt.ylabel('MAE', fontsize=12, fontweight='bold')
# plt.xlabel('Epochs', fontsize=12, fontweight='bold')
# plt.legend(["Train", "Validation"], loc="upper left")
# plt.savefig("Overfit.png")


def no_dropout_model(hp):
    model = Sequential()
    model.add(
        Dense(
            kernel_initializer='normal',
            units=hp.Int("units", min_value=8, max_value=512, step=8),
            activation='relu',
        )
    )
    model.add(
        Dense(
            units=hp.Int("units", min_value=8, max_value=512, step=8),
            activation='relu',
        )
    )
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(
            hp.Choice("learning_rate", values=[1e-3, 1e-5, 1e-7, 1e-9]),
        ),
        loss="mse",
        metrics=["mse", "mae"],
    )
    return model


def dropout_model(hp):
    model = Sequential()
    model.add(
        Dense(
            kernel_initializer='normal',
            units=hp.Int("units", min_value=8, max_value=512, step=8),
            activation='relu',
        )
    )
    model.add(
        Dense(
            units=hp.Int("units", min_value=8, max_value=512, step=8),
            activation='relu',
        )
    )
    model.add(
        Dropout(
            hp.Float(
                'dropout',
                min_value=0.0,
                max_value=0.1,
                default=0.005,
                step=0.01)
        )
    )
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(
            hp.Choice("learning_rate", values=[1e-3, 1e-5, 1e-7, 1e-9]),
        ),
        loss="mse",
        metrics=["mse", "mae"],
    )
    return model


# tuner = RandomSearch(
#     build_model,
#     objective="val_mse",
#     max_trials=5,
#     executions_per_trial=5,
#     overwrite=True,
#     directory="temp"
# )

tuner = Hyperband(
    no_dropout_model,
    objective="val_mse",
    max_epochs=100,
    hyperband_iterations=1,
    project_name="untitled_project"
)

# tuner.search_space_summary()
tuner.search(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val),
             callbacks=[tf.keras.callbacks.TensorBoard('untitled_project')])

# model: Sequential() = tuner.get_best_models(num_models=5)[0]

# tuner.results_summary()


# pred_results = model.predict(x_train)

# pva = DataFrame(y_test, columns=['pred_actual'])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
rmtree("untitled_project")

model = tuner.hypermodel.build(best_hps)

history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
plt.plot(history.history['mae'], '-*')
plt.plot(history.history['val_mae'], '--o')
plt.title('MAE: Train and Validation', fontsize=14, fontweight='bold')
plt.ylabel('MAE', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.legend(["Train", "Validation"], loc="upper left")
plt.savefig("Overfit.png")

pva = None
r2 = []
for i in range(5):
    model = tuner.hypermodel.build(best_hps)
    model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_val, y_val))
    predictions = model.predict(x_test)
    r2.append(r2_score(y_test, predictions))
    if i == 0:
        pva = DataFrame(predictions, columns=[f"pred_{i}"])
    else:
        pva[f"pred_{i}"] = predictions

pva["pred_std"] = pva.std(axis=1)
pva["pred_avg"] = pva.mean(axis=1)
pva["actual"] = y_test

r2 = np.array(r2)
r2_std = r2.std()
r2_avg = r2.mean()

prediction_stats = dict(
    r2_avg=r2_avg,
    r2_std=r2_std,
    mse_avg=mean_squared_error(pva['actual'], pva['pred_avg']),
    rmse_avg=np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg'])),
    pred_std=pva["pred_std"],
    pred_avg=pva["pred_avg"],
)

graph.pva_graph(prediction_stats, pva, "dev", scaled=False)
# print(history)
