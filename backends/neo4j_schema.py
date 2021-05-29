from pandas import DataFrame
from pathlib import Path
from ast import literal_eval
from warnings import warn
from typing import Union
from tqdm import tqdm

from neo4j_backends import PseudoNode, PseudoRelationship, Gather, __drop_none_prop_values__


class ReadSchema:

    def __init__(self, schema_file: str):
        # Set some default neo4j param stuff
        self.uri: str = "bolt://localhost:7687"
        self.auth: tuple = ("neo4j", "password")
        self.apply_constraints: bool = True
        self.gathered: Union[Gather, None] = None
        self.query: str = ""

        self.file: str = schema_file
        self.lines: list[str] = self.gather_sig_lines()
        if not self.lines:
            return
        self.separated_lines: list[list[str]] = self.separate_lines()
        self.holding_nodes: dict[str, dict[str, Union[str, dict[str, str], list[str]]]] = self.parse_nodes()
        self.holding_relationships: list[dict] = self.parse_relationships()
        self.bulk: bool = self.determine_bulk()

    def gather_sig_lines(self):
        lines = []
        with open(self.file, 'r') as f:
            for line in f:
                if line.find("#") != -1:
                    line = line.split("#")[0]
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    def separate_lines(self):
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
    def parse_prop(prop_dict: dict, prop: str):
        merge_prop = False
        general_prop = False
        unique_prop = False
        lonely_prop = False
        strict_prop = False
        while True:
            prop = prop.strip()
            if prop[-1] == '*':  # unique prop
                unique_prop = True
                prop = prop[:-1]
            elif prop[-1] == '!':  # lonely prop
                lonely_prop = True
                prop = prop[:-1]
            elif prop.split(":")[1].strip()[0] == "{" and prop.split(":")[1].strip()[-1] == "}":  # strict prop
                strict_prop = True
                prop = prop.replace("{", "")
                prop = prop.replace("}", "")
            else:
                break

        if prop[:2] == "--":  # Merge Prop
            merge_prop = True
            prop = prop[2:]
        elif prop[:1] == "-":  # General Prop
            general_prop = True
            prop = prop[1:]

        neo4j_key, df_column = prop.split(":")[0].strip(), prop.split(":")[1].strip()

        if merge_prop:
            prop_dict['merge_props'][neo4j_key] = df_column
        elif general_prop:
            prop_dict['general_props'][neo4j_key] = df_column

        if unique_prop:
            prop_dict['unique_props'].append(neo4j_key)
        if lonely_prop:
            prop_dict['lonely_props'].append(neo4j_key)
        if strict_prop:
            prop_dict['strict_props'].append(neo4j_key)

        return prop_dict

    def parse_props(self, entity):
        # Gather property information
        node_props_dict = {'merge_props': {}, 'general_props': {},
                           'unique_props': [], 'lonely_props': [],
                           'strict_props': []}
        for prop in entity:
            node_props_dict = self.parse_prop(node_props_dict, prop)
        return node_props_dict

    def parse_nodes(self):
        nodes = {}
        for entity in self.separated_lines:
            entity_type = entity[0].split("|")[0]
            if entity_type.lower() == "node":
                # Gather information in node header
                node_id = entity[0].split("|")[1]
                node_name = entity[0].split("|")[2]
                entity.pop(0)  # First line is always header
                # Gather property information
                node_props_dict = self.parse_props(entity)
                nodes[node_id] = {'name': node_name}
                nodes[node_id].update(node_props_dict)
        return nodes

    def parse_relationships(self):
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
                entity.pop(0)  # First line is always header
                # Gather property information
                rel_props_dict = self.parse_props(entity)
                relationships.append({"name": rel_name, "node_1": rel_node_1, "direction": direction,
                                      "node_2": rel_node_2, "merge_props": rel_props_dict['merge_props'],
                                      "general_props": rel_props_dict['general_props']})
        return relationships

    def determine_bulk(self):
        bulk = True
        for node, node_props in self.holding_nodes.items():
            if not node_props['merge_props']:
                bulk = False
            if node_props['lonely_props']:
                bulk = False
        return bulk

    def merge_bulk(self, df):
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
        self.gathered = Gather(nodes, rels, bulk=True, uri=self.uri, auth=self.auth,
                               apply_constraints=self.apply_constraints)
        self.query = self.gathered.query
        self.gathered.merge(df)

    def merge(self, df: DataFrame = None, uri: str = "bolt://localhost:7687",
              auth: tuple = ("neo4j", "password"), apply_constraints: bool = True, bulk: bool = None,
              suppress_warning: bool = False):

        self.uri = uri
        self.auth = auth
        self.apply_constraints = apply_constraints

        # Check if bulk pass in matches self.bulk, if not warn user
        if not suppress_warning:
            if not self.bulk and bulk:
                warn("Dynamic nodes detected, but bulk was set to True. If the code fails, make sure each node has a "
                     "merge property, and that ! was not passed. Or, set bulk to False. If you wish to not see this "
                     "warning, pass suppress_warning=True")
            if self.bulk and not bulk and bulk is not None:
                warn("No dynamic nodes detected, but bulk was set to False. This will cause the code to run much "
                     "slower. If you wish to not see this warning, pass suppress_warning=True")
        if bulk is not None:
            self.bulk = bulk

        if self.bulk:
            self.merge_bulk(df)
            return

        # TODO add logic for non-bulk


if __name__ == "__main__":
    file = str(Path(__file__).parent.parent / "files/other/example.schema")
    rows = [{'first name': "Fred", 'last name': 'Smith', 'hair color': "brown", 'age': 35, 'country': 'USA',
             'tip': 1150, 'lived in for years': 35, 'clothing': 'Supreme', 'clothing type': 'Hoodie'},
            {'first name': "Leo", 'last name': 'Johnson', 'hair color': "blonde", 'age': 42, 'country': 'USA',
             'tip': 1150, 'lived in for years': 42, 'clothing': '', 'clothing type': ''},
            {'first name': "John", 'last name': 'Walker', 'hair color': "brown", 'age': 25, 'country': 'USA',
             'tip': 1151, 'lived in for years': 23, 'clothing': '', 'clothing type': 'shirt'}
            ]
    data = DataFrame(rows)
    schema_obj = ReadSchema(schema_file=file)
    schema_obj.merge(df=data, bulk=False, suppress_warning=True)
