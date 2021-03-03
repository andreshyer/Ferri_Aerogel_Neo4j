import pandas as pd


def cleanup(file):
    df = pd.read_csv(file)
    df['Index'] = df.index
    rows = []
    for index, row in df.iterrows():
        row = dict(row)
        for item, value in row.items():
            if value == "----":
                row[item] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(file, index=False)


if __name__ == "__main__":
    cleanup("Aerogel.csv")
