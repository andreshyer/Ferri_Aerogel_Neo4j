import pandas as pd

import neo4j_backends

if __name__ == "__main__":
    rows = [{'first name': "Fred", 'last name': 'Smith', 'hair color': "brown", 'age': 35, 'country': 'USA',
             'tip': 1150, 'lived in for years': 35},
            {'first name': "Leo", 'last name': 'Johnson', 'hair color': "blonde", 'age': 42, 'country': 'USA',
             'tip': 1150, 'lived in for years': 42}]
    data = pd.DataFrame(rows)
    nodes = []
    relationships = []

    Fred = neo4j_backends.PseudoNode(node_name="Person",
                                     merge_props={'first_name': 'first name', 'last_name': "last name"},
                                     general_props={'hair_color': 'hair color', 'age': 'age'},
                                     bulk=True)
    nodes.append(Fred)

    usa = neo4j_backends.PseudoNode(node_name="Country",
                                    merge_props={'name': 'country'},
                                    unique_prop_keys=['name'],
                                    bulk=True)
    nodes.append(usa)

    rel_1 = neo4j_backends.PseudoRelationship("LivesIn", Fred, "-", usa,
                                              merge_props={'tip': 'tip'},
                                              general_props={"lived_in_for_years": 'lived in for years'})
    relationships.append(rel_1)

    gathered = neo4j_backends.Gather(nodes, relationships)
    print(gathered.query)
    gathered.merge(data=data)
