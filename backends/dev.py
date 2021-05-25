import pandas as pd
from pathlib import Path

from neo4j_backends import insert_from_schema

if __name__ == "__main__":
    rows = [{'first name': "Fred", 'last name': 'Smith', 'hair color': "brown", 'age': 35, 'country': 'USA',
             'tip': 1150, 'lived in for years': 35, 'clothing': 'Supreme', 'clothing type': 'Hoodie'},
            {'first name': "Leo", 'last name': 'Johnson', 'hair color': "blonde", 'age': 42, 'country': 'USA',
             'tip': 1150, 'lived in for years': 42, 'clothing': '', 'clothing type': ''}]
    data = pd.DataFrame(rows)

    schema_file = str(Path(__file__).parent.parent / "files/example.schema")
    insert_from_schema(schema_file=schema_file, df=data)
