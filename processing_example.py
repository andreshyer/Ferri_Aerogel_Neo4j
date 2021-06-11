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

    injester = Ingester(data, columns_to_drop=drop_columns)
    injester.replace_compounds_with_smiles()
    injester.replace_nan_with_zeros()
    data = injester.replace_words_with_numbers(ignore_smiles=False)

    featurizer = Featurizer(data)  # TODO add logic to actually featurize the data
    featurizer.featurize_molecules(method=['rdkit2d'])

    complex_processor = ComplexDataProcessor(df=data, y_columns=y_columns)
    data = complex_processor.get_only_important_columns()

    splitter = DataSplitter(df=data, y_columns=y_columns,
                            train_percent=0.8, test_percent=0.2, val_percent=0,
                            grouping_column=paper_id_column, state=None)
    x_test, x_train, x_val, y_test, y_train, y_val = splitter.split_data()
