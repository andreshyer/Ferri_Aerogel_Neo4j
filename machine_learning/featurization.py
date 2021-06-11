
from pandas import DataFrame
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

from machine_learning import Ingester


class Featurizer(Ingester):

    def __init__(self, df: DataFrame):
        """
        This is the class that will take the SMILES from ingest and featurize them. The logic still needs to be
        added here.

        :param df: Pandas DataFrame
        """
        super().__init__(df=df)

    def featurize_molecules(self, method: list[str]):
        """
        Actual function that featurizes the data

        :param method: descriptor name from descriptastorus
        :return: Pandas DataFrame
        """
        # self.__test_col_if_obj_smiles__()
        generator = MakeGenerator(("rdkit2d",))

        def do_featurize(smi):
            data = generator.process(smi)

        columns = []
        for name, numpy_type in generator.GetColumns():
            columns.append(name)
        self.df[columns] = self
        data = generator.process(smiles="c1ccccc1")
