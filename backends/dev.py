import pandas as pd
import requests
import json
from ast import literal_eval
from pathlib import Path
from re import match
from numpy import nan
from tqdm import tqdm


if __name__ == "__main__":
    """
    Baldy coded stuff just to help format the data files into machine readable files
    """

    # schema_example()

    cached_compound_info = pd.read_csv('cached_compound_info.csv')

    for index, row in tqdm(cached_compound_info.iterrows()):
        compound = row['compound']
        if str(row['smiles']) == "nan":

            cid = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound}/cids/TXT").text
            cid = cid.split()[0]

            # Fetch JSON data of compound
            response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON").text
            data = json.loads(response)
            with open('dev.json', "w") as f:
                json.dump(data, f, indent=6)

            if isinstance(data, dict):
                if 'Fault' in data.keys():
                    pass
                else:
                    sections = data['Record']['Section']
                    for section in sections:
                        if section['TOCHeading'] == "Names and Identifiers":
                            section = section['Section']
                            for subsection in section:
                                if subsection['TOCHeading'] == "Computed Descriptors":
                                    try:
                                        smiles = subsection['Section'][3]['Information'][0]['Value']
                                        smiles = smiles['StringWithMarkup'][0]['String']
                                        cached_compound_info.at[index, 'smiles'] = smiles
                                        cached_compound_info.at[index, 'cid'] = cid
                                    except IndexError:
                                        pass
            cached_compound_info.to_csv('dev.csv')

    # data_file = str(Path(__file__).parent.parent / "files/si_aerogels/si_aerogel_machine_readable.csv")
    # df: pd.DataFrame = pd.read_csv(data_file)
    # df = df.dropna(axis=1, how="all")

    # for index, row in df.iterrows():
    #     row = dict(row)
    #     for key, value in row.items():
    #         if "," in str(value):
    #             if len(value.split(",")) != len(value.split(", ")):
    #                 print(value)
    # df.to_csv(str(Path(__file__).parent.parent / "files/si_aerogels/si_aerogel_machine_readable.csv"))
    #
    # for col in df.columns:
    #
    #     if col != "Final Material":
    #         try:
    #             df[col].astype(float)
    #         except ValueError:
    #             length = 1
    #             for value in df[col].tolist():
    #                 if str(value).find(", ") != -1:
    #                     if len(value.split(", ")) > length:
    #                         length = len(value.split(", "))
    #             if length > 1:
    #                 new_columns_values = []
    #                 for i in range(1, length+1):
    #                     new_columns_values.append([])
    #                 for value in df[col].tolist():
    #                     if str(value) == 'nan':
    #                         split_values = [""] * length
    #                     else:
    #                         split_values = str(value).split(", ")
    #                         split_values.extend([""] * (length - len(split_values)))
    #                     for index, sub_value in enumerate(split_values):
    #                         new_columns_values[index].append(sub_value)
    #
    #                 for i in range(len(new_columns_values)):
    #                     new_col_name = f"{col} ({i})"
    #                     df[new_col_name] = new_columns_values[i]
    #                 df = df.drop([col], axis=1)
    # df.to_csv(str(Path(__file__).parent.parent / "files/si_aerogels/dev.csv"), index=False)

    # data = str(Path(__file__).parent.parent / "files/si_aerogels/si_aerogel_machine_readable.csv")
    # data = pd.read_csv(data)
    # titles = data['Title'].unique()
    #
    # indexes = dict(zip(titles, range(len(titles))))
    #
    # def find_index(x):
    #     return indexes[x]
    #
    # data['paper_id'] = data['Title'].apply(find_index)
    # data.to_csv('dev.csv')
