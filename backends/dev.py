import pandas as pd
from pathlib import Path

from neo4j_backends import insert_from_schema


def schema_example():
    rows = [{'first name': "Fred", 'last name': 'Smith', 'hair color': "brown", 'age': 35, 'country': 'USA',
             'tip': 1150, 'lived in for years': 35, 'clothing': 'Supreme', 'clothing type': 'Hoodie'},
            {'first name': "Leo", 'last name': 'Johnson', 'hair color': "blonde", 'age': 42, 'country': 'USA',
             'tip': 1150, 'lived in for years': 42, 'clothing': '', 'clothing type': ''},
            {'first name': "John", 'last name': 'Walker', 'hair color': "brown", 'age': 25, 'country': 'USA',
             'tip': 1151, 'lived in for years': 23, 'clothing': '', 'clothing type': 'shirt'}
            ]
    data = pd.DataFrame(rows)

    schema_file = str(Path(__file__).parent.parent / "files/other/example.schema")
    insert_from_schema(schema_file=schema_file, df=data)


if __name__ == "__main__":
    schema_example()

    # Baldy coded stuff just to help format the data files into machine readable files
    # data_file = str(Path(__file__).parent.parent / "files/si_aerogels/dev.csv")
    # df: pd.DataFrame = pd.read_csv(data_file)
    # df = df.replace("----", "")
    # df = df.replace("Ambient", 0.101325)
    # df = df.dropna(axis=1, how="all")
    #
    # new_values = []
    # col_name = "Surface Area (m2/g)"
    # for value in df[col_name].tolist():
    #     if str(value) == "nan":
    #         new_values.append("")
    #     else:
    #         if len(str(value).split(", ")) > 1:
    #             value = (float(str(value).split(", ")[0]) + float(str(value).split(", ")[1])) / 2
    #         new_values.append(float(value))
    # df[col_name] = new_values
    # df.to_csv(str(Path(__file__).parent.parent / "files/si_aerogels/dev.csv"), index=False)
    #
    # for col in df.columns:
    #     try:
    #         df[col].astype(float)
    #     except ValueError:
    #         length = 1
    #         for value in df[col].tolist():
    #             if str(value).find(", ") != -1:
    #                 if len(value.split(", ")) > length:
    #                     length = len(value.split(", "))
    #         if length > 1:
    #             new_columns_values = []
    #             for i in range(1, length+1):
    #                 new_columns_values.append([])
    #             for value in df[col].tolist():
    #                 if str(value) == 'nan':
    #                     split_values = [""] * length
    #                 else:
    #                     split_values = str(value).split(", ")
    #                     split_values.extend([""] * (length - len(split_values)))
    #                 for index, sub_value in enumerate(split_values):
    #                     new_columns_values[index].append(sub_value)
    #
    #             for i in range(len(new_columns_values)):
    #                 new_col_name = f"{col} <{i}>"
    #                 df[new_col_name] = new_columns_values[i]
    #             df = df.drop([col], axis=1)
    # df.to_csv(str(Path(__file__).parent.parent / "files/si_aerogels/si_aerogel_machine_readable.csv"), index=False)

