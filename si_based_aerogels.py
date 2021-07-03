from pandas import read_excel

from backends import ReadSchema


if __name__ == "__main__":
    port = 'bolt://localhost:7687'
    database = 'neo4j'
    username = 'neo4j'
    password = 'password'
    df = read_excel('files/si_aerogels/dev.xlsx')

    schema = "files/si_aerogels/si_aerogel_schema.schema"

    schema_obj = ReadSchema(schema_file=schema)
    schema_obj.merge(df=df, uri=port, database=database,
                     auth=(username, password), apply_constraints=True, bulk=True)
