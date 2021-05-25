from pandas import read_csv

from backends.neo4j_backends import insert_from_schema


if __name__ == "__main__":
    port = 'bolt://localhost:7687'
    username = 'neo4j'
    password = 'password'
    df = read_csv('files/si_aerogels/si_aerogels.csv')
    schema = "files/si_aerogels/si_aerogel.schema"

    insert_from_schema(schema_file=schema, df=df, uri=port,
                       auth=(username, password), apply_constraints=True)
