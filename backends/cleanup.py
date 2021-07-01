import pandas as pd
from numpy import nan

def cleanup(file):
    df = pd.read_excel(file)
    df = df.dropna(how='all', axis=0)
    df['Index'] = df.index
    rows = []
    for index, row in df.iterrows():
        row = dict(row)
        for item, value in row.items():
            if value == "----":
                row[item] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel(file, index=False)


def cleanup_dataframe(df: pd.DataFrame):

    def __ttcn__(n):  # Cleans up data to be inserted into neo4j
        if not n:
            return n
        if str(n) == "nan":
            return None
        try:
            return float(n)
        except ValueError:
            return n
        except TypeError:
            print(n)
            raise TypeError

    rows = df.where(df.notnull(), None)
    rows = rows.to_dict('records')
    new_rows = []
    for row in rows:
        new_row = {}
        for key, value in row.items():
            if value == "----":
                new_row[key] = nan
            else:
                new_row[key] = __ttcn__(value)
        new_rows.append(new_row)
    return pd.DataFrame(new_rows)


if __name__ == "__main__":
    cleanup("si_aerogels.xlsx")
