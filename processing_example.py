from pathlib import Path

import pandas as pd

from machine_learning import Ingester, DataSplitter, ComplexDataProcessor, Featurizer


if __name__ == "__main__":
    data = pd.read_csv(str(Path(__file__).parent / "files/si_aerogels/si_aerogel_AI_machine_readable.csv"))
    y_columns = ['Surface Area (m2/g)', 'Thermal Conductivity (W/mK)']
    # y_columns = ['Surface Area (m2/g)']
    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                    'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase', 'Nanoparticle Size (nm)',
                    'Average Pore Size (nm)']
    paper_id_column = "paper_id"

    featurizer = Featurizer(data, columns_to_drop=drop_columns)
    featurizer.remove_non_smiles_str_columns()
    featurizer.replace_compounds_with_smiles()
    featurizer.featurize_molecules(method='rdkit2d')
    data = featurizer.replace_nan_with_zeros()

    # complex_processor = ComplexDataProcessor(df=data, y_columns=y_columns)
    # feature_importances, important_columns = complex_processor.get_only_important_columns(number_of_models=5)
    # data = data[important_columns]

    splitter = DataSplitter(df=data, y_columns=y_columns,
                            train_percent=0.8, test_percent=0.2, val_percent=0,
                            grouping_column=paper_id_column, state=None)
    x_test, x_train, x_val, y_test, y_train, y_val = splitter.split_data()
