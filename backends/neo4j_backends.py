import string
from tqdm import tqdm
from warnings import warn
from ast import literal_eval

from pandas import DataFrame
from neo4j import GraphDatabase


def __drop_none_prop_values__(prop_dict: dict, prop_dict_type: str = "merge", supress_warning: bool = False):
    """
    Simple function that looks at the prop_dict and drop any property keys that have a corresponding property value
    of None. Neo4j will normally drop these values on its own, but it is done here to be able to catch errors that
    would be hard to nail down otherwise.

    :param prop_dict: the dict containing the property key and property values
    :param prop_dict_type: Whether prop_dict is from merge props or general props
    :return:
    """

    if not prop_dict:
        return prop_dict

    new_prop_dict = {}
    for prop_key, prop_value in prop_dict.items():
        if prop_value:
            new_prop_dict[prop_key] = prop_value
        if isinstance(prop_value, int) and prop_value == 0:
            new_prop_dict[prop_key] = prop_value

    if not supress_warning:
        if prop_dict_type == "merge" and len(new_prop_dict) != len(prop_dict):
            warn("Some property values in the merge prop dict are None for a Node."
                 "This may lead to duplicate nodes in Neo4j.")

    return new_prop_dict


def __format_merge_props__(merge_props_dict: dict, bulk: bool, class_type: str = 'Node'):
    """
    The merge prop dict can be inserted directly in as a string for a Neo4j Cypher. But, for efficiency, the
    UNWIND function is used, so property values are replace with $prop_key, where the property values are pass
    in from a prop_dict.

    This function just does a bit of clean up.

    :return: merge props as a string
    """
    if not merge_props_dict and class_type != 'Node':
        return ""

    formatted_string = " {"
    for prop_key, prop_value in merge_props_dict.items():
        if bulk:
            formatted_string += f"`{prop_key}`: row.`{prop_value}`, "
        else:
            if not isinstance(prop_value, int) and not isinstance(prop_value, float):
                formatted_string += f'`{prop_key}`: "{prop_value}", '
            else:
                formatted_string += f"`{prop_key}`: {prop_value}, "
    formatted_string = formatted_string[:-2] + "}"  # Drop trailing comma and space and add closing bracket
    return formatted_string


def __format_general_props__(query_letter: str, general_props_dict: dict, bulk: bool):
    """
    While the merged properties are formatted fine for Neo4j Cypher queries, mostly. The general properties are
    a bit more picky. A general Cypher query to merge a node will look like

    MERGE (n: Person {name: Ales})
        ON CREATE SET n.eye_color = brown, n.hair_color = brown

    Which can be rewritten in terms of the class variable

    MERGE (n: <self.name> <self.__merge_props_str__>)
        ON CREATE SET n.<general_prop_key_1> = <general_prop_value_1>,
            n.<general_prop_key_2> = <general_prop_value_2>, ..., n.<general_prop_key_n> = <general_prop_key_n>

    This function will format that ON CREATE SET part of the cypher query

    :return: general props as a string
    """
    # If general props is empty, return empty string
    if not general_props_dict:
        return ""
    # Else create a ON CREATE SET string and add the prop keys and values to it
    formatted_string = "ON CREATE SET"
    for prop_key, prop_val in general_props_dict.items():
        if bulk:
            formatted_string += f" {query_letter}.`{prop_key}` = row.`{prop_val}`,"
        else:
            if not isinstance(prop_val, int) and not isinstance(prop_val, float) and not isinstance(prop_val, list):
                formatted_string += f' {query_letter}.`{prop_key}` = "{prop_val}",'
            else:
                formatted_string += f" {query_letter}.`{prop_key}` = {prop_val},"
    formatted_string = formatted_string[:-1]  # Drop trailing comma
    return formatted_string


def __convert_n_to_letters__(n: int):
    """
    Simple code that converts a number > 0 into letters similar to how excel labels columns.
    0 = a, 1 = b, ... 25 = z, 26 = aa, 27 = ab, 28 = ac, ....

    The source for this code can be found at
    https://stackoverflow.com/questions/48983939/convert-a-number-to-excel-s-base-26

    :param n: Integer equal to or greater than 0
    :return: Letters corresponding to the number passed
    """
    if n < 0:
        raise TypeError(f"n must be equal to or greater than 0, a value of {n} was passed")
    n += 1

    def div_mod_excel(i):
        a, b = divmod(i, 26)
        if b == 0:
            return a - 1, b + 26
        return a, b

    chars = []
    while n > 0:
        n, d = div_mod_excel(n)
        chars.append(string.ascii_lowercase[d - 1])
    return ''.join(reversed(chars))


class PseudoNode:

    def __init__(self, node_name: str, merge_props: dict, general_props: dict = None,
                 unique_prop_keys: list[str] = None):
        """
        This class is not an actual node class, like py2neo, but rather holds the information needed to merge and
        create relationships. All information that is passed into Neo4j is passed through the Neo4j driver.

        The property dicts should be formatted as

        non-bulk: {'<property key to be inserted into Neo4j>': <property value>}

        or

        bulk: {'<property key to be inserted into Neo4j>': row.`<property value>`} where <property value> is
            the key name in the list of dicts being passed

        :param node_name: The name of the Node Type to be merged into Neo4j
        :param merge_props: The properties that the node is to be merged with. While Neo4j does not demand that
            nodes be merged, this code does. So, there must be some unique key that a node is merged on. It is highly
            recommended to have an Index value if no other key exist for a node to be merged on.
        :param general_props: The other properties that should not be merged into Neo4j
        :param unique_prop_keys: The property keys that constraints should be applied to
        """

        # Verify properties were correctly passed into class
        if not node_name:
            raise TypeError("Name of Node must be specified")
        if not isinstance(node_name, str):
            raise TypeError("Name of Node must be a string")
        if not merge_props:
            raise TypeError("Merge props must be specified.")
        if not isinstance(merge_props, dict):
            raise TypeError("Merge props must be a dict")
        if not isinstance(general_props, dict) and general_props:
            raise TypeError("General props must either be a dict or None")

        self.node_name: str = node_name
        self.merge_props_dict: dict = __drop_none_prop_values__(merge_props, "merge")
        self.general_props_dict: dict = __drop_none_prop_values__(general_props, "general")
        self.unique_prop_keys: list[str] = unique_prop_keys

        # Verify all values were not dropped in the values to be merged
        if not self.merge_props_dict:
            raise TypeError("All property values in merge props are none, at least one must be specified")


class PseudoRelationship:

    def __init__(self, rel_name: str, a: PseudoNode, rel_direction: str, b: PseudoNode, merge_props: dict = None,
                 general_props: dict = None):
        """
        Just as with the PseudoNode, this is not an actual relationship object like py2neo, but just holds the
        information needed to create the relationship.

        Most of the properties here mirror the PseudoNode class, with a few key exceptions.

        a is the first PseudoNode in the relationship and b is the second, with the rel_direction defining how
        they relate. The options for rel_direction are

        "->": (a)-[]->(b)
        "<-": (b)-[]->(a)
        "-": (a)-[]-(b) : No direction

        Then a tuple is created with this information
        self.__rel__ = (left node: PseudoNode, right node: PseudoNode, direction: str)

        :param rel_name: The name of the relationship being merged
        :param a: PseudoNode 1
        :param rel_direction: The direction of the relationship
        :param b: PseudoNode 2
        :param merge_props: The properties that the relationship is to be merged with.
        :param general_props: The other properties that should not be merged into Neo4j
        """

        # Verify properties were correctly passed into class
        if not isinstance(merge_props, dict) and merge_props:
            raise TypeError("Merge props must be a dict or None")
        if not isinstance(general_props, dict) and general_props:
            raise TypeError("General props must either be a dict or None")

        self.rel_name: str = rel_name
        self.a: PseudoNode = a
        self.b: PseudoNode = b
        self.merge_props_dict: dict = __drop_none_prop_values__(merge_props, "merge")
        self.general_props_dict: dict = __drop_none_prop_values__(general_props, "general")

        # Check what direction the relationship is going, if any, and create relationship tuple
        if rel_direction == "->":
            self.__rel__: tuple = (a, b, ">")
        elif rel_direction == "<-":
            self.__rel__: tuple = (b, a, ">")
        elif rel_direction == "-":
            self.__rel__: tuple = (a, b, "")
        else:
            raise TypeError("Rel direction must be either '->', '<-', or '-'")


class Gather:

    def __init__(self, nodes: list[PseudoNode], relationships: list[PseudoRelationship], bulk: bool = False,
                 uri: str = "bolt://localhost:7687", auth: tuple = ("neo4j", "password"),
                 apply_constraints: bool = True):
        """
        This goal of this class is to take all the nodes and relationships, and create a query that can be passed into
        Neo4j. Also, the driver connection to Neo4j is not created until the function self.merge() is called. So, if
        only the query is desired, then the Neo4j graph does not need to be running.

        :param nodes: List of PseudoNodes
        :param relationships: List of PseudoRelationships
        :param uri: uri to the Neo4j graph, recommended bolt connection
        :param auth: auth tuple, default ("neo4j", "password")
        :param apply_constraints: bool to decide if constraints should be applied to nodes and relationships
        """

        self.nodes: list[PseudoNode] = nodes
        self.relationships: list[PseudoRelationship] = relationships
        self.uri: str = uri
        self.auth: tuple[str, str] = auth
        self.bulk: bool = bulk
        self.__constraints__: bool = apply_constraints

        # These are not true indexes, but rather letters are used to correlate nodes and relations in Neo4j Cypher
        self.__indexed_nodes__: dict[PseudoNode, str] = self.__index_nodes__()
        self.__indexed_relationships__: dict[PseudoRelationship, str] = self.__index_relationships__()
        self.query: str = self.__generate_query__()

    def __index_nodes__(self):
        indexes = {}
        for n, node in enumerate(self.nodes):
            indexes[node] = __convert_n_to_letters__(n)
        return indexes

    def __index_relationships__(self):
        indexes = {}
        for n, relationship in enumerate(self.relationships):
            n += len(self.nodes)  # Want to start where the indexes for the nodes left off
            indexes[relationship] = __convert_n_to_letters__(n)
        return indexes

    def __generate_query__(self):
        """
        This code is fairly difficult to follow on its own. The easiest way to show how this code works is by showing
        some examples.

        Say you have a person named Fred Smith who has brown hair and is aged 35. He has also lived in the USA all his
        life. And, there was some tip ID number where this information came from. A query to insert this data into
        Neo4j would look like.

        MERGE (a:Person {first_name: 'Fred', last_name: 'Smith'})
            ON CREATE SET a.hair_color = 'brown', a.age = 35
        MERGE (b:Country {name: 'USA'})

        MERGE (a)-[c:LivesIn {tip: 1150}]->(b)
            ON CREATE SET c.lived_in_for_years = 35

        Where the first section are the nodes to be merged into Neo4j, and second are the relationships
        If data is being inserted in bulk, and some example data looks like.

          first name last name hair color  age country   tip  lived in for years
        0       Fred     Smith      brown   35     USA  1150                  35
        1        Leo   Johnson     blonde   42     USA  1150                  42

        Then the correlating query would look like

        UNWIND $rows as row

        MERGE (a:Person {first_name: row.`first name`, last_name: row.`last name`})
            ON CREATE SET a.hair_color = row.`hair color`, a.age = row.`age`
        MERGE (b:Country {name: row.`country`})

        MERGE (a)-[c:LivesIn {tip: row.`tip`}]->(b)
            ON CREATE SET c.lived_in_for_years = row.`lived in for years`

        Where the $rows are just the rows from data.to_dict('records'). It is assumed that the data
        being passed in are pandas DataFrames.

        So, to goal of this function is to format the information being passed in into the queries above. While
        this exact format is by no means needed, this format is used to try and be as readable as possible.

        :return: None
        """

        # Generate query header
        if self.bulk:
            query = "UNWIND $rows as row\n"
        else:
            query = ""

        # Generate node section
        for node in self.nodes:
            merge_props_str = __format_merge_props__(node.merge_props_dict, self.bulk, class_type='Node')
            line = f"\nMERGE ({self.__indexed_nodes__[node]}:{node.node_name}{merge_props_str})"
            general_props_str = __format_general_props__(self.__indexed_nodes__[node],
                                                         node.general_props_dict, self.bulk)
            if general_props_str:
                line += f"\n    {general_props_str}"
            query += line

        query += "\n"  # Separate Node and Relationship sections

        # Generate relationship section
        for relationship in self.relationships:
            # (left_node_index)-[rel_index:rel_name {rel_merge props}]-(<right_node_index>)
            merge_props_str = __format_merge_props__(relationship.merge_props_dict, self.bulk,
                                                     class_type='Relationship')
            line = f"\nMERGE ({self.__indexed_nodes__[relationship.__rel__[0]]})-"
            line += f"[{self.__indexed_relationships__[relationship]}:"
            line += f"{relationship.rel_name}{merge_props_str}]-{relationship.__rel__[2]}"
            line += f"({self.__indexed_nodes__[relationship.__rel__[1]]})"
            # Set general props
            general_props_str = __format_general_props__(self.__indexed_relationships__[relationship],
                                                         relationship.general_props_dict, self.bulk)
            if general_props_str:
                line += f"\n    {general_props_str}"
            query += line

        # Drop any leading or trailing spaces
        query = query.strip()
        return query

    def merge(self, data: DataFrame = None):
        """
        This takes all the nodes and relationships passed into the Gather class, and merges them into Neo4j. If bulk
        was set to True for the nodes, then the pandas dataframe with data should be passed as well.

        :param data: DataFrame of data
        :return: None
        """
        # TODO limit the amount of rows being inserted when bulk inserting

        if self.__constraints__:
            self.__apply_constraints__()

        def __insert_data__(tx, query):
            tx.run(query, rows=rows)

        rows = None
        if isinstance(data, DataFrame):
            if not data.empty:
                rows = data.to_dict('records')
        if self.bulk and not isinstance(data, DataFrame):
            raise Exception("bulk was set to true, but no data was passed")

        driver = GraphDatabase.driver(self.uri, auth=self.auth)
        with driver.session() as session:
            session.write_transaction(__insert_data__, self.query)

    def __apply_constraints__(self):
        """
        All this function does is parse all the unique property keys in the nodes and relationships, then applies
        constraints on those keys.

        Unfortunately, allowing node keys requires enterprise edition of neo4j. While local install this would be
        fine, for remote database, this will break.

        :return: None
        """

        def __insert_constraint__(tx):
            tx.run(query)

        driver = GraphDatabase.driver(self.uri, auth=self.auth)
        with driver.session() as session:
            for node in self.nodes:
                if node.unique_prop_keys:
                    for unique_prop_key in node.unique_prop_keys:
                        query = f"""
                        CREATE CONSTRAINT IF NOT EXISTS ON (n:{node.node_name}) ASSERT n.{unique_prop_key} IS UNIQUE
                        """.strip()
                        session.write_transaction(__insert_constraint__)


def insert_from_schema(schema_file: str, df: DataFrame, uri: str = "bolt://localhost:7687",
                       auth: tuple = ("neo4j", "password"), apply_constraints: bool = True,
                       force_non_bulk: bool = False):
    """
    The purpose of this code is to take a schema file and DataFrame and insert the information into Neo4j. The code
    for this does not make sense outside of the context of example.schema. The file can be found under files and
    explains the logic. This code matches that logic from the schema file.

    :param schema_file: Schema that defines nodes and relationships
    :param df: DataFrame with data that corresponds to schema file
    :param uri: uri to the Neo4j graph, recommended bolt connection
    :param auth: auth tuple, default ("neo4j", "password")
    :param apply_constraints: bool to decide if constraints should be applied to nodes and relationships
    :param force_non_bulk: Force each row to be inserted without bulk insert
    :return: None
    """
    # TODO look for errors in schema file
    # TODO implement logic for ! in schema files

    # Grab only significant lines, drop all comments
    lines = []
    with open(schema_file, 'r') as file:
        for line in file:
            if line.find("#") != -1:
                line = line.split("#")[0]
            line = line.strip()
            if line:
                lines.append(line)

    # If readable schema is empty, return None
    if not lines:
        return

    # Group lines into Nodes and Relationships
    list_of_separated_lines: list[list[str]] = []
    separated_lines: list[str] = []
    for line in lines:
        if line.find("|") != -1:
            list_of_separated_lines.append(separated_lines)
            separated_lines = [line]
        else:
            separated_lines.append(line)
    list_of_separated_lines.append(separated_lines)
    list_of_separated_lines.pop(0)  # First list always is blank

    # Parse info in separated lines
    schema_nodes = []
    schema_relationships = []
    for separated_lines in list_of_separated_lines:
        merge_props = {}
        general_props = {}
        unique_keys = []
        header = separated_lines[0]  # First line in separated lines is always header line
        entity_type = header.split("|")[0].strip()
        entity_id = header.split("|")[1].strip()  # If relationship, this is relationship details
        entity_name = header.split("|")[2].strip()
        for i in range(1, len(separated_lines)):  # Gather properties, start at next line after header
            separated_line = separated_lines[i]
            if separated_line.find("--") != -1:  # Merge property
                if separated_line.strip()[-1] == "*":
                    separated_line = separated_line.strip("*")[:-1].strip()
                    unique_keys.append(separated_line.split(":")[0].split("--")[1].strip())
                separated_line = separated_line.split("--")[1]
                merge_props[separated_line.split(":")[0].strip()] = separated_line.split(":")[1].strip()
            else:
                separated_line = separated_line.split("-")[1]  # General property
                general_props[separated_line.split(":")[0].strip()] = separated_line.split(":")[1].strip()
        if entity_type == 'Node':
            schema_nodes.append({"id": entity_id, "node_name": entity_name, "merge_props": merge_props,
                                 "general_props": general_props, "unique_keys": unique_keys})
        elif entity_type == 'Rel':  # Relationships need a bit more work
            if entity_id.find("->") != -1:  # Relationships dont have an id, parse info from this section
                direction = "->"
                node_1 = entity_id.split("->")[0].strip()
                node_2 = entity_id.split("->")[1].strip()
            else:
                direction = "-"
                node_1 = entity_id.split("-")[0].strip()
                node_2 = entity_id.split("-")[1].strip()
            schema_relationships.append({"rel_name": entity_name, "node_1": node_1, "direction": direction,
                                         "node_2": node_2, "merge_props": merge_props, "general_props": general_props})

    # check if nodes and relationships can be inserted in bulk
    if force_non_bulk:
        bulk = False
    else:
        bulk = True
        for schema_node in schema_nodes:
            if not schema_node['merge_props']:
                bulk = False

    # Insert data as bulk if possible, requires all nodes to have a unique key
    if bulk:
        node_key = {}
        nodes = []
        rels = []
        for schema_node in schema_nodes:
            node = PseudoNode(schema_node['node_name'], merge_props=schema_node['merge_props'],
                              general_props=schema_node['general_props'], unique_prop_keys=schema_node['unique_keys'])
            nodes.append(node)
            node_key[schema_node['id']] = node
        for rel in schema_relationships:
            rel = PseudoRelationship(rel['rel_name'], node_key[rel['node_1']], rel['direction'],
                                     node_key[rel['node_2']], merge_props=rel['merge_props'],
                                     general_props=rel['general_props'])
            rels.append(rel)
        gathered = Gather(nodes, rels, bulk=True, uri=uri, auth=auth, apply_constraints=apply_constraints)
        gathered.merge(df)

    # If cannot insert data as bulk, insert data one row at a time (much, much slower)
    # Data is parsed from each row, and a query is generated that is inserted into Neo4j
    else:

        # Helper function to replace key values with actual values from dataframe
        def rvip(prop_dict, row_data):  # replace values in prop dict
            new_prop_dict = {}
            for key, value in prop_dict.items():
                if value[0] == "{" and value[-1] == "}":
                    value = value[1:-1]
                    value = literal_eval(value)
                    new_prop_dict[key] = value
                else:
                    value = row_data[value]
                    if str(value) == "nan":
                        value = None
                    new_prop_dict[key] = value
            return new_prop_dict

        index_column = df.index.values.tolist()
        df['index'] = index_column  # Create column with indexes
        for row in tqdm(df.to_dict('records'), total=len(df), desc="Inserting data into Neo4j"):
            node_key = {}
            nodes = []
            rels = []

            # Insert nodes
            for schema_node in schema_nodes:
                merge_props = __drop_none_prop_values__(rvip(schema_node['merge_props'], row), supress_warning=True)
                general_props = __drop_none_prop_values__(rvip(schema_node['general_props'], row), supress_warning=True)

                # Make sure to only add index to merge props if general_props
                if not merge_props and general_props:
                    merge_props['index'] = row['index']
                    schema_node['unique_keys'].append('index')

                if merge_props or general_props:
                    node = PseudoNode(schema_node['node_name'], merge_props=merge_props,
                                      general_props=general_props,
                                      unique_prop_keys=schema_node['unique_keys'])
                    nodes.append(node)
                    node_key[schema_node['id']] = node

            # Insert relationships
            for rel in schema_relationships:
                merge_props = __drop_none_prop_values__(rvip(rel['merge_props'], row),
                                                        supress_warning=True)
                general_props = __drop_none_prop_values__(rvip(rel['general_props'], row),
                                                          supress_warning=True)
                if rel['node_1'] in node_key.keys() and rel['node_2'] in node_key.keys():
                    rel = PseudoRelationship(rel['rel_name'], node_key[rel['node_1']], rel['direction'],
                                             node_key[rel['node_2']], merge_props=merge_props,
                                             general_props=general_props)
                    rels.append(rel)
            gathered = Gather(nodes, rels, bulk=False, uri=uri, auth=auth, apply_constraints=apply_constraints)
            gathered.merge()
