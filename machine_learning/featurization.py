
from pandas import DataFrame
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from rdkit.Chem import MolFromSmiles

from machine_learning import Ingester


class Featurizer(Ingester):

    def __init__(self, df: DataFrame):
        """
        This is the class that will take the SMILES from ingest and featurize them. The logic still needs to be
        added here.

        :param df: Pandas DataFrame
        """

        self.raw_df: DataFrame = df
        self.df: DataFrame = df
        super().__init__(df=self.df)

    def featurize_molecules(self, method: list[str]):
        """
        Actual function that featurizes the data

        :param method: descriptor name from descriptastorus
        :return: Pandas DataFrame
        """
        # self.__test_col_if_obj_smiles__()

        generator = MakeGenerator(("rdkit2d",))
        columns = []
        for name, numpy_type in generator.GetColumns():
            columns.append(name)
        data = generator.process(smiles="c1ccccc1")
