import string
from tqdm import tqdm
from warnings import warn
from tqdm import tqdm
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


def __format_merge_props__(merge_props_dict: dict[str, str], bulk: bool, class_type: str = 'Node'):
    """
    The merge prop dict can be inserted directly in as a string for a Neo4j Cypher. But, for efficiency, the
    UNWIND function is used, so property values are replace with $prop_key, where the property values are pass
    in from a prop_dict.

    This function just does a bit of clean up.

    :return: merge props as a string
    """
    if not merge_props_dict and class_type != 'Node':
        return None, ""

    df_column_name = list(merge_props_dict.values())[0]  # TODO Allow this for multiple merge props
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
    return df_column_name, formatted_string


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

        self.df_column, _ = __format_merge_props__(self.merge_props_dict, class_type="Node", bulk=True)


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
                 uri: str = "bolt://localhost:7687", database: str = "neo4j",
                 auth: tuple = ("neo4j", "password"), apply_constraints: bool = True):
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
        self.database: str = database
        self.auth: tuple[str, str] = auth
        self.bulk: bool = bulk
        self.__constraints__: bool = apply_constraints

        # These are not true indexes, but rather letters are used to correlate nodes and relations in Neo4j Cypher
        self.__indexed_nodes__: dict[PseudoNode, str] = self.__index_nodes__()
        self.__indexed_relationships__: dict[PseudoRelationship, str] = self.__index_relationships__()
        if self.bulk:
            self.nodes_query, self.rels_query = self.__generate_bulk_query__()  # Just one query needed for bulk
        else:
            self.queries: list[str] = self.__generate_non_bulk_queries__()

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

    def __generate_bulk_query__(self):
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
        nodes_query = "UNWIND $rows as row"

        # Generate node section
        for node in self.nodes:
            node_index = self.__indexed_nodes__[node]
            df_column_name, merge_props_str = __format_merge_props__(node.merge_props_dict, self.bulk,
                                                                     class_type='Node')
            general_props_str = __format_general_props__(self.__indexed_nodes__[node],
                                                         node.general_props_dict, self.bulk)
            line = "\nWITH row"
            line += f"\n    WHERE NOT row.`{df_column_name}` IS NULL"
            line += f"\n    MERGE ({node_index}: {node.node_name}{merge_props_str})"
            if general_props_str:
                line += f"\n    {general_props_str}"
            line += f"\n"

            nodes_query += line
        nodes_query = nodes_query.strip()

        rels_query = "UNWIND $rows as row"  # Separate Node and Relationship sections

        # Generate relationship section
        for relationship in self.relationships:
            # (left_node_index)-[rel_index:rel_name {rel_merge props}]-(<right_node_index>)
            left_node_index = self.__indexed_nodes__[relationship.__rel__[0]]
            left_node_node_name = relationship.__rel__[0].node_name
            left_node_df_col, \
            left_node_merge_props = __format_merge_props__(relationship.__rel__[0].merge_props_dict,
                                                           self.bulk, class_type='Node')

            rel_index = self.__indexed_relationships__[relationship]
            rel_name = relationship.rel_name
            direction = relationship.__rel__[2]
            formatted_string, merge_props_str = __format_merge_props__(relationship.merge_props_dict, self.bulk,
                                                                       class_type='Relationship')
            general_props_str = __format_general_props__(self.__indexed_relationships__[relationship],
                                                         relationship.general_props_dict, self.bulk)

            right_node_index = self.__indexed_nodes__[relationship.__rel__[1]]
            right_node_node_name = relationship.__rel__[1].node_name
            right_node_df_col, \
            right_node_merge_props = __format_merge_props__(relationship.__rel__[1].merge_props_dict,
                                                            self.bulk, class_type='Node')

            line = "\nWITH row"
            line += f"\n    WHERE NOT row.`{left_node_df_col}` IS NULL"
            line += f"\n    AND NOT row.`{right_node_df_col}` IS NULL"
            line += f"\n    MATCH ({left_node_index}: {left_node_node_name}{left_node_merge_props})"
            line += f"\n    MATCH ({right_node_index}: {right_node_node_name}{right_node_merge_props})"
            line += f"\n    MERGE ({left_node_index})-[{rel_index}: {rel_name}{merge_props_str}]-{direction}({right_node_index})"
            if general_props_str:
                line += f"\n    {general_props_str}"
            line += "\n"
            rels_query += line
        rels_query = rels_query.strip()

        return nodes_query, rels_query

    def __generate_non_bulk_queries__(self):

        queries = []

        for node in self.nodes:
            merge_props_str = __format_merge_props__(node.merge_props_dict, self.bulk, class_type='Node')
            query = f"\nMERGE ({self.__indexed_nodes__[node]}:{node.node_name}{merge_props_str})"
            general_props_str = __format_general_props__(self.__indexed_nodes__[node],
                                                         node.general_props_dict, self.bulk)
            if general_props_str:
                query += f"\n    {general_props_str}"
            queries.append(query)

        for rel in self.relationships:
            rel_id = self.__indexed_relationships__[rel]
            merge_props_str = __format_merge_props__(rel.merge_props_dict, self.bulk, class_type='Relationship')
            general_props_str = __format_general_props__(rel_id, rel.general_props_dict, self.bulk)
            rel_name = rel.rel_name

            rel = rel.__rel__
            direction = rel[2]
            node_1_id = self.__indexed_nodes__[rel[0]]
            node_1_name = rel[0].node_name
            node_1_merge_props = __format_merge_props__(rel[0].merge_props_dict, self.bulk, class_type='Node')
            node_2_id = self.__indexed_nodes__[rel[1]]
            node_2_name = rel[1].node_name
            node_2_merge_props = __format_merge_props__(rel[1].merge_props_dict, self.bulk, class_type='Node')

            query = f"""
            MATCH ({node_1_id}: {node_1_name}{node_1_merge_props})
            MATCH ({node_2_id}: {node_2_name}{node_2_merge_props})
            
            MERGE ({node_1_id})-[{rel_id}: {rel_name}{merge_props_str}]-{direction}({node_2_id})
            {general_props_str}
            """
            queries.append(query)

        return queries

    def merge(self, data: DataFrame = None, batch: int = 1000):
        """
        This takes all the nodes and relationships passed into the Gather class, and merges them into Neo4j. If bulk
        was set to True for the nodes, then the pandas dataframe with data should be passed as well.

        :param data: DataFrame of data
        :param batch: Number of rows to insert into Neo4j at one time when merging with bulk. Lower this number
                        if RAM is an issue
        :return: None
        """
        # TODO limit the amount of rows being inserted when bulk inserting

        if self.__constraints__:
            self.__apply_constraints__()

        def __insert_data__(tx, query):
            tx.run(query, rows=rows)

        # Verify params were pass correctly
        rows = None
        if isinstance(data, DataFrame):
            if not data.empty:
                rows = data.to_dict('records')
            else:
                raise Exception("bulk was set to true, but no data was passed")

        driver = GraphDatabase.driver(self.uri, auth=self.auth)

        # Insert queries if not bulk
        if not self.bulk:
            with driver.session(database=self.database) as session:
                for query in self.queries:
                    session.write_transaction(__insert_data__, query)
            return

        print("Inserting Data into Neo4j")

        # Insert all data if bulk
        i = 0
        rows_to_merge = []
        with driver.session(database=self.database) as session:
            for row in rows:
                rows_to_merge.append(row)
                i += 1
                if i % batch == 0:
                    session.write_transaction(__insert_data__, self.nodes_query)
                    session.write_transaction(__insert_data__, self.rels_query)
                    rows_to_merge = []
                    i = 0
            if rows_to_merge:
                session.write_transaction(__insert_data__, self.nodes_query)
                session.write_transaction(__insert_data__, self.rels_query)

    def __apply_constraints__(self):
        """
        All this function does is parse all the unique property keys in the nodes and relationships, then applies
        constraints on those keys.

        Unfortunately, allowing node keys requires enterprise edition of neo4j. While local install this would be
        fine, for remote database, this will break.

        :return: None
        """

        print("Applying Constraints")

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
