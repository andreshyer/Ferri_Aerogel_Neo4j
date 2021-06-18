from pandas import DataFrame
from pathlib import Path
from ast import literal_eval
from warnings import warn
from typing import Optional
from tqdm import tqdm

from .neo4j_backends import PseudoNode, PseudoRelationship, Gather, __drop_none_prop_values__


class ReadSchema:

    def __init__(self, schema_file: str):
        # Set some default neo4j param stuff
        self.uri: str = "bolt://localhost:7687"
        self.basebase = "neo4j"
        self.auth: tuple = ("neo4j", "password")
        self.apply_constraints: bool = True
        self.gathered: Optional[Gather] = None
        self.query: str = ""

        print("Parsing Schema File")

        # Parse information from schema file
        self.file: str = schema_file
        self.lines: list[str] = self.__gather_sig_lines__()
        if not self.lines:
            return
        self.separated_lines: list[list[str]] = self.__separate_lines__()
        self.holding_nodes: dict[str, dict] = self.__parse_nodes__()
        self.holding_relationships: list[dict] = self.__parse_relationships__()
        self.bulk: bool = self.__determine_bulk__()

    def __gather_sig_lines__(self):
        lines = []
        with open(self.file, 'r') as f:
            for line in f:
                if line.find("#") != -1:
                    line = line.split("#")[0]
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    def __separate_lines__(self):
        list_of_separated_lines: list[list[str]] = []
        separated_lines: list[str] = []
        for line in self.lines:
            if line.find("|") != -1:
                list_of_separated_lines.append(separated_lines)
                separated_lines = [line]
            else:
                separated_lines.append(line)
        list_of_separated_lines.append(separated_lines)
        list_of_separated_lines.pop(0)  # First list always is blank
        return list_of_separated_lines

    @staticmethod
    def __parse_prop__(prop_dict: dict, prop: str):
        # Purposely avoiding regex for this, trying to avoid some buffer issues

        # Holding variables
        merge_prop = False
        general_prop = False
        unique_prop = False
        strict_prop = False

        # Look for tags
        while True:
            prop = prop.strip()
            if prop[-1] == '*':  # unique prop
                unique_prop = True
                prop = prop[:-1]
            elif "{" in prop and "}" in prop:
                if prop.split(":")[1].strip()[0] == "{" and prop.split(":")[1].strip()[-1] == "}":  # strict prop
                    strict_prop = True
                    prop = prop.replace("{", "")
                    prop = prop.replace("}", "")
            else:
                break

        # Determine if merge or general prop
        if prop[:2] == "--":  # Merge Prop
            merge_prop = True
            prop = prop[2:]
        elif prop[:1] == "-":  # General Prop
            general_prop = True
            prop = prop[1:]

        # Collect neo4j property key and DataFrame column with value (or strict value)
        neo4j_key, df_column = prop.split(":")[0].strip(), prop.split(":")[1].strip()
        if merge_prop:
            prop_dict['merge_props'][neo4j_key] = df_column
        elif general_prop:
            prop_dict['general_props'][neo4j_key] = df_column

        # Tack on additional prop information
        if unique_prop:
            prop_dict['unique_props'].append(neo4j_key)
        if strict_prop:
            prop_dict['strict_props'].append(neo4j_key)

        return prop_dict

    def __parse_props__(self, entity):
        prop_dict = {'merge_props': {}, 'general_props': {},
                     'unique_props': [], 'strict_props': []}
        for prop in entity:
            prop_dict = self.__parse_prop__(prop_dict, prop)
        return prop_dict

    def __parse_nodes__(self):
        nodes = {}
        for entity in self.separated_lines:
            entity_type = entity[0].split("|")[0]
            if entity_type.lower() == "node":
                # Gather information in node header
                node_id = entity[0].split("|")[1]
                node_name = entity[0].split("|")[2]
                entity.pop(0)  # First line is always header

                # Gather property information
                node_props_dict = self.__parse_props__(entity)

                # Create holding node
                nodes[node_id] = {'name': node_name}
                nodes[node_id].update(node_props_dict)
        return nodes

    def __parse_relationships__(self):
        relationships = []
        for entity in self.separated_lines:
            entity_type = entity[0].split("|")[0]
            if entity_type.lower() == "rel":
                # Gather information in relationship header
                rel_rel: str = entity[0].split("|")[1]
                if "->" in rel_rel:
                    direction = "->"
                    rel_node_1 = rel_rel.split("->")[0]
                    rel_node_2 = rel_rel.split("->")[1]
                elif "<-" in rel_rel:
                    direction = "->"
                    rel_node_1 = rel_rel.split("->")[1]
                    rel_node_2 = rel_rel.split("->")[0]
                else:
                    direction = "-"
                    rel_node_1 = rel_rel.split("-")[0]
                    rel_node_2 = rel_rel.split("-")[1]
                rel_name = entity[0].split("|")[2]
                header = entity.pop(0)  # First line is always header

                # Gather property information
                rel_props_dict = self.__parse_props__(entity)

                # Create holding relationship, note that this drops unique props
                relationships.append({"name": rel_name, "node_1": rel_node_1, "direction": direction,
                                      "node_2": rel_node_2, "merge_props": rel_props_dict['merge_props'],
                                      "general_props": rel_props_dict['general_props'],
                                      "strict_props": rel_props_dict['strict_props']})

                # Warn user that * and ! tags were dropped in relationships if found
                if rel_props_dict['unique_props']:
                    warn(f"* tag detected on relationship {header}. The * tag has been ignored.")
        return relationships

    def __determine_bulk__(self):
        """
        Simply check to see if information should be merged in bulk or not. If any node does not have a merge property
        or has any properties with ! tag, define nodes at dynamic nodes. Dynamic nodes should not be bulk inserted and
        will likely break the code if tried.

        :return:
        """
        bulk = True
        for node, node_props in self.holding_nodes.items():
            if not node_props['merge_props']:
                bulk = False
        return bulk

    def __merge_bulk__(self, df: DataFrame, batch: int):
        node_key = {}
        nodes = []
        rels = []
        for holding_node_id, holding_node in self.holding_nodes.items():
            node = PseudoNode(holding_node['name'], merge_props=holding_node['merge_props'],
                              general_props=holding_node['general_props'],
                              unique_prop_keys=holding_node['unique_props'])
            nodes.append(node)
            node_key[holding_node_id] = node
        for holding_rel in self.holding_relationships:
            holding_rel = PseudoRelationship(holding_rel['name'], node_key[holding_rel['node_1']],
                                             holding_rel['direction'],
                                             node_key[holding_rel['node_2']],
                                             merge_props=holding_rel['merge_props'],
                                             general_props=holding_rel['general_props'])
            rels.append(holding_rel)
        self.gathered = Gather(nodes, rels, bulk=True, uri=self.uri, database=self.database,
                               auth=self.auth, apply_constraints=self.apply_constraints)
        self.gathered.merge(df, batch=batch)

    def __merge_non_bulk__(self, df: DataFrame):

        def __pseudo_literal_eval__(value):
            if str(value).strip() == "nan":
                return None
            elif str(value).strip() == "":
                return None
            elif str(value).strip() == "None":
                return None
            else:
                try:
                    value = float(value)
                    if int(value) == float(value):
                        value = int(value)
                    return value
                except ValueError:
                    try:
                        value = literal_eval(value)  # Mainly looking for list objects
                    except ValueError:
                        return value
                    except SyntaxError:
                        return value

        def __gather_values__(entity_type: str, entity: dict, row_data):
            merge_props = {}
            for neo4j_key, df_column in entity['merge_props'].items():
                if entity_type == 'node' and neo4j_key in entity['strict_props']:
                    value = df_column
                else:
                    value = row_data[df_column]
                value = __pseudo_literal_eval__(value)
                merge_props[neo4j_key] = value
            merge_props = __drop_none_prop_values__(merge_props, supress_warning=True)

            general_props = {}
            for neo4j_key, df_column in entity['general_props'].items():
                if neo4j_key in entity['strict_props']:
                    value = df_column
                else:
                    value = row_data[df_column]
                value = __pseudo_literal_eval__(value)
                general_props[neo4j_key] = value
            general_props = __drop_none_prop_values__(general_props, supress_warning=True)

            return merge_props, general_props

        def sub_main(data):
            data['index'] = data.index.tolist()
            data = data.to_dict('records')
            for row in tqdm(data, total=len(data), desc="Inserting data into Neo4j"):
                node_key = {}
                nodes = []
                for neo4j_key, node in self.holding_nodes.items():
                    merge_props, general_props = __gather_values__('node', node, row)

                    # Make sure to only add index to merge props if general_props
                    if not merge_props and general_props:
                        merge_props['index'] = row['index']
                        node['unique_props'].append('index')

                    if merge_props or general_props:
                        node = PseudoNode(node['name'], merge_props=merge_props,
                                          general_props=general_props,
                                          unique_prop_keys=node['unique_props'])
                        nodes.append(node)
                        node_key[neo4j_key] = node

                relationships = []
                for relationship in self.holding_relationships:
                    if relationship['node_1'] in node_key.keys() and relationship['node_2'] in node_key.keys():
                        merge_props, general_props = __gather_values__('rel', relationship, row)
                        rel = PseudoRelationship(relationship['name'],
                                                 node_key[relationship['node_1']],
                                                 relationship['direction'],
                                                 node_key[relationship['node_2']],
                                                 merge_props=merge_props,
                                                 general_props=general_props)
                        relationships.append(rel)

                gathered = Gather(nodes, relationships, bulk=False, uri=self.uri, database=self.database,
                                  auth=self.auth, apply_constraints=self.apply_constraints)
                gathered.merge()

        sub_main(df)

    def merge(self, df: DataFrame, uri: str = "bolt://localhost:7687", database: str = "neo4j",
              auth: tuple = ("neo4j", "password"), apply_constraints: bool = True, bulk: bool = None,
              suppress_warning: bool = False, batch: int = 1000):

        self.uri = uri
        self.database = database
        self.auth = auth
        self.apply_constraints = apply_constraints

        # Check if bulk passed in matches self.bulk, if not warn user
        if not suppress_warning:
            if not self.bulk and bulk:
                warn("Dynamic nodes detected, but bulk was set to True. If the code fails, make sure each node has a "
                     "merge property, and that ! was not passed. Or, set bulk to False. If you wish to not see this "
                     "warning, pass suppress_warning=True")
            if self.bulk and not bulk and bulk is not None:
                warn("No dynamic nodes detected, but bulk was set to False. This will cause the code to run much "
                     "slower. But, this is useful if certain nodes have a merge property that can be np.nan or None in "
                     "the DataFrame. If you wish to not see this warning, pass suppress_warning=True")
        if bulk is not None:
            self.bulk = bulk

        if self.bulk:
            self.__merge_bulk__(df, batch)
        else:
            self.__merge_non_bulk__(df)


if __name__ == "__main__":
    file = str(Path(__file__).parent.parent / "files/other/example.schema")
    rows = [{'first name': "Fred", 'last name': 'Smith', 'hair color': "brown", 'age': 35, 'country': 'USA',
             'tip': 1150, 'lived in for years': 35, 'clothing': 'Supreme', 'clothing type': 'Hoodie'},
            {'first name': "Leo", 'last name': 'Johnson', 'hair color': "blonde", 'age': 42, 'country': 'USA',
             'tip': 1150, 'lived in for years': 42, 'clothing': '', 'clothing type': ''},
            {'first name': "John", 'last name': 'Walker', 'hair color': "brown", 'age': 25, 'country': "UK",
             'tip': 1151, 'lived in for years': 23, 'clothing': '', 'clothing type': 'shirt'}
            ]
    people_countries = DataFrame(rows)
    schema_obj = ReadSchema(schema_file=file)
    schema_obj.merge(df=people_countries, bulk=True)
