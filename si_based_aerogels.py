import pandas as pd
from tqdm import tqdm

from py2neo import Graph, Node, Relationship, RelationshipMatcher


class AerogelsToNeo4j:

    def __init__(self, port, username, password, data):
        self.graph = Graph(port, username=username, password=password)
        self.data = pd.read_csv(data)
        self.__merge_data__()

    @staticmethod
    def __ttcn__(n):  # Cleans up data to be inserted into neo4j
        if not n:
            return n
        if str(n) == "nan":
            return None
        try:
            return float(n)
        except ValueError:
            return n

    def __merge_data__(self):
        for index, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Inserting data into Neo4j"):
            row = dict(row)

            """
            This first section defines the core nodes in the knowledge graph. Each Final gel has different
            things defined, so to avoid null nodes certain nodes are dropped.
            
            Required Nodes:
                - Final gel
                - Non-dried gel
                - Drying method
            """

            # Define Final Gel Node
            final_gel_props = dict(
                final_material=self.__ttcn__(row['Final Material']),
                general_notes=self.__ttcn__(row['Notes']),
                pore_volume=self.__ttcn__(row['Pore Volume (cm3/g)']),
                average_pore_diameter=self.__ttcn__(row['Average Pore Diameter (nm)']),
                nanoparticle_size=self.__ttcn__(row['Nanoparticle Size (nm)']),
                surface_area=self.__ttcn__(row['Surface Area (m2/g)']),
                bulk_density=self.__ttcn__(row['Bulk Density (g/cm3)']),
                thermal_conductivity=self.__ttcn__(row['Thermal Conductivity (W/mK)']),
                young_modulus=self.__ttcn__(row['Young Modulus (MPa)']),
                drying_notes=self.__ttcn__(row['Drying Notes'])
            )
            final_gel = Node("FinalGel", **final_gel_props)
            self.graph.merge(final_gel, "FinalGel", "final_material")

            # Define Non-dried gel
            non_dired_gel_props = dict(
                id=row['Index'],
                anna_notes=self.__ttcn__(row["Anna's Notes"]),
                final_sol_pH=self.__ttcn__(row['pH final sol']),
                gelation_temp=self.__ttcn__(row['Gelation Temp (°C)']),
                gelation_pressure=self.__ttcn__(row['Gelation Pressure (MPa)']),
                gelation_time=self.__ttcn__(row['Gelation Time (mins)']),
                hydrolysis_time=self.__ttcn__(row['Hydrolysis Time (hr)']),
                aging_conditions=self.__ttcn__(row['Aging Conditions']),
                aging_temp=self.__ttcn__(row['Aging Temp (°C)']),
                aging_time=self.__ttcn__(row['Aging Time (hrs)']),
                gelation_washing_notes=self.__ttcn__(row['Gelation/Washing Notes'])
            )
            non_dried_gel = Node("NonDriedGel", **non_dired_gel_props)
            self.graph.merge(non_dried_gel, "NonDriedGel", "id")

            # Define Drying Method Node
            drying_method_props = dict(
                method=self.__ttcn__(row['Drying Method'])
            )
            drying_method = Node("DryingMethod", **drying_method_props)
            self.graph.merge(drying_method, "DryingMethod", "method")

            # Define Sintering Node
            sintering_props = dict(
                notes=self.__ttcn__(row['Sintering Notes']),
                atmosphere=self.__ttcn__(row['Sintering Atmosphere'])
            )
            if any(sintering_props.values()):
                sintering_props['id'] = row['Index']
                sintering = Node("Sintering", **sintering_props)
                self.graph.merge(sintering, "Sintering", "id")
            else:
                sintering = None

            # Define Crystalline Phase Node
            if self.__ttcn__(row['Crystalline Phase']):
                crystalline_props = dict(
                    name=self.__ttcn__(row['Crystalline Phase'])
                )
                crystalline_phase = Node("CrystallinePhase", **crystalline_props)
                self.graph.merge(crystalline_phase, "CrystallinePhase", 'name')
            else:
                crystalline_phase = None

            # Define Porosity Node
            if self.__ttcn__(row['Porosity']):
                porosity_props = dict(
                    name=self.__ttcn__(row['Porosity'])
                )
                porosity = Node("Porosity", **porosity_props)
                self.graph.merge(porosity, "Porosity", 'name')
            else:
                porosity = None

            # Define lit info Node
            lit_info_props = dict(
                title=self.__ttcn__(row['Title']),
                year=self.__ttcn__(row['Year']),
                citied_references=self.__ttcn__(row['Cited References (#)']),
                timed_citied=self.__ttcn__(row['Times Cited (#)'])
            )
            lit_info = Node("LitInfo", **lit_info_props)
            self.graph.merge(lit_info, 'LitInfo', 'title')

            def __parse_authors__(raw_authors, corresponding):
                authors = []
                if raw_authors:
                    raw_authors = str(raw_authors).split(", ")
                    for raw_author in raw_authors:
                        rap = raw_author.split("(")
                        if len(rap) == 2:
                            author = rap[0]
                            email = rap[1][:-1]
                        else:
                            author = raw_author
                            email = None
                        author_props = dict(
                            name=author,
                            email=email,
                            corresponding_author=corresponding
                        )
                        author = Node("Author", **author_props)
                        authors.append(author)
                return authors

            authors = []

            # Define non-corresponding authors
            raw_authors = self.__ttcn__(row['Authors'])
            authors.extend(__parse_authors__(raw_authors, corresponding=False))

            # Define corresponding authors
            raw_authors = self.__ttcn__(row['Corresponding Author'])
            authors.extend(__parse_authors__(raw_authors, corresponding=True))

            for author in authors:
                self.graph.merge(author, "Author", "name")

            # Define formation method
            formation_method_props = dict(
                method=self.__ttcn__(row['Formation Method'])
            )
            if formation_method_props['method']:
                formation_method = Node("FormationMethod", **formation_method_props)
                self.graph.merge(formation_method, "FormationMethod", "method")
            else:
                formation_method = None

            # Define sol nodes
            sol_1_props = dict(
                compontents=self.__ttcn__(row['Sol 1']),
                stir_rate=self.__ttcn__(row['Sol 1 Stir rate (rpm)']),
                stir_time=self.__ttcn__(row['Sol 1 Stir Time (min)']),
                temp=self.__ttcn__(row['Sol 1 Temp (°C)']),
            )
            if any(sol_1_props.values()):
                sol_1_props['id'] = row['Index']
                sol_1 = Node("Sol", **sol_1_props)
                self.graph.merge(sol_1, "Sol", "id")
            else:
                sol_1 = None

            sol_2_props = dict(
                compontents=self.__ttcn__(row['Sol 2']),
                stir_rate=self.__ttcn__(row['Sol 2 Stir rate (rpm)']),
                stir_time=self.__ttcn__(row['Sol 2 Stir Time (min)']),
                temp=self.__ttcn__(row['Sol 2 Temp (°C)']),
            )
            if any(sol_2_props.values()):
                sol_2_props['id'] = row['Index']
                sol_2 = Node("Sol", **sol_2_props)
                self.graph.merge(sol_2, "Sol", "id")
            else:
                sol_2 = None

            # Define washing solvents
            wash_solvent_1_props = dict(
                name=self.__ttcn__(row['Wash Solvent 1']),

            )
            if wash_solvent_1_props['name']:
                wash_solvent_1 = Node("Solvent", **wash_solvent_1_props)
                self.graph.merge(wash_solvent_1, "Solvent", "name")
            else:
                wash_solvent_1 = None

            wash_solvent_2_props = dict(
                name=self.__ttcn__(row['Wash Solvent 2']),

            )
            if wash_solvent_2_props['name']:
                wash_solvent_2 = Node("Solvent", **wash_solvent_2_props)
                self.graph.merge(wash_solvent_2, "Solvent", "name")
            else:
                wash_solvent_2 = None

            wash_solvent_3_props = dict(
                name=self.__ttcn__(row['Wash Solvent 3']),

            )
            if wash_solvent_3_props['name']:
                wash_solvent_3 = Node("Solvent", **wash_solvent_3_props)
                self.graph.merge(wash_solvent_3, "Solvent", "name")
            else:
                wash_solvent_3 = None

            wash_solvent_4_props = dict(
                name=self.__ttcn__(row['Wash Solvent 4']),

            )
            if wash_solvent_4_props['name']:
                wash_solvent_4 = Node("Solvent", **wash_solvent_4_props)
                self.graph.merge(wash_solvent_4, "Solvent", "name")
            else:
                wash_solvent_4 = None

            """
            This section defines the parameter nodes in the knowledge graph.
            
            There are no required nodes for this section.
            
            More than just the nodes are being defined here. Since both the non-dried and final gels share the
            same parameter nodes, all parameter nodes are stored in a tuple. Where the first item in the tuple is the
            node with the properties on the node. And the second item are the properties for the relationship between
            the parameter node and the gel nodes. 
            
            Also, all parameter nodes are stored in a dict. It would work to store them in a list, but having them
            in a dict might make coding easier in the future. 
            """

            # Define parameter nodes dict
            parameter_nodes = {}

            # Define solvents
            solvent_1_props = dict(
                name=self.__ttcn__(row['Solvent 1'])
            )
            if solvent_1_props['name']:
                solvent_1 = Node("Solvent", **solvent_1_props)
                self.graph.merge(solvent_1, "Solvent", "name")
                parameter_nodes["solvent_1"] = (solvent_1,
                                                {'concentration': self.__ttcn__(row['Solvent 1 Concentration (M)'])})

            solvent_2_props = dict(
                name=self.__ttcn__(row['Solvent 2'])
            )
            if solvent_2_props['name']:
                solvent_2 = Node("Solvent", **solvent_2_props)
                self.graph.merge(solvent_2, "Solvent", "name")
                parameter_nodes["solvent_2"] = (solvent_2,
                                                {'concentration': self.__ttcn__(row['Solvent 2 Concentration (M)'])})

            # Define modifier
            modifier_props = dict(
                name=self.__ttcn__(row['Modifier'])
            )
            if modifier_props['name']:
                modifier = Node("Modifier", **modifier_props)
                self.graph.merge(modifier, "Modifier", "name")
                parameter_nodes['modifier'] = (modifier,
                                               {'concentration': self.__ttcn__(row['Modifier Solvent (M)'])})
            else:
                modifier = None

            modifier_solvent_props = dict(
                name=self.__ttcn__(row['Modifier Solvent'])
            )
            if modifier_solvent_props['name']:
                modifier_solvent = Node("Solvent", **modifier_solvent_props)
                self.graph.merge(modifier_solvent, "Solvent", "name")
            else:
                modifier_solvent = None

            # Define gelation agent
            gelation_agent_props = dict(
                name=self.__ttcn__(row['Gelation Agent'])
            )
            if gelation_agent_props['name']:
                gelation_agent = Node("GelationAgent", **gelation_agent_props)
                self.graph.merge(gelation_agent, "GelationAgent", "name")
                parameter_nodes['gelation_agent'] = (gelation_agent,
                                                     {"concentration": self.__ttcn__(row['Gelation Agent (M)']),
                                                      "temp": self.__ttcn__(row['Gelation Temp (°C)'])})

            # Define Dopant
            dopant_props = dict(
                name=self.__ttcn__(row['Dopant'])
            )
            if dopant_props['name']:
                dopant = Node("Dopant", **dopant_props)
                self.graph.merge(dopant, "Dopant", "name")
                parameter_nodes['dopant'] = (dopant,
                                             {'concentration': self.__ttcn__(row['Dopant Concentration (M)'])})

            # Define acid catalyst
            acid_catalyst_props = dict(
                name=self.__ttcn__(row['Acid Catalyst'])
            )
            if acid_catalyst_props['name']:
                acid_catalyst = Node("AcidCatalyst", **acid_catalyst_props)
                self.graph.merge(acid_catalyst, "AcidCatalyst", "name")
                parameter_nodes['acid_catalyst'] = (acid_catalyst,
                                                    {"initial concentration": self.__ttcn__(
                                                        row['Acid Catalyst Initial Concentration (M)']),
                                                        "concentration in sol": self.__ttcn__(
                                                            row['Acid Catalyst concentration in Sol(M)'])})

            # Define base catalyst
            base_catalyst_props = dict(
                name=self.__ttcn__(row['Base Catalyst'])
            )
            if base_catalyst_props['name']:
                base_catalyst = Node("BaseCatalyst", **base_catalyst_props)
                self.graph.merge(base_catalyst, "BaseCatalyst", "name")
                parameter_nodes['base_catalyst'] = (base_catalyst,
                                                    {"initial concentration": self.__ttcn__(
                                                        row['Base Catalyst Initial Concentration(M)']),
                                                     "concentration in sol": self.__ttcn__(
                                                         row['Base Catalyst concentration in Sol (M)'])})

            # Define Si precursor
            si_precursor_props = dict(
                name=self.__ttcn__(row['Si Precursor'])
            )
            if si_precursor_props['name']:
                si_precursor = Node("SiPrecursor", **si_precursor_props)
                self.graph.merge(si_precursor, "SiPrecursor", "name")
                parameter_nodes['si_precursor'] = (si_precursor,
                                                   {"concentration": self.__ttcn__(
                                                       row['Si Precursor Concentration (M)'])})

            # Define Hybrid precursor
            hybrid_precursor_props = dict(
                name=self.__ttcn__(row['Hybrid Aerogel Co-Precursor'])
            )
            if hybrid_precursor_props['name']:
                hybrid_precursor = Node("HybridPrecursor", **hybrid_precursor_props)
                self.graph.merge(hybrid_precursor, "HybridPrecursor", "name")
                parameter_nodes['hybrid_precursor'] = (hybrid_precursor,
                                                       {'concentration': self.__ttcn__(
                                                           row['Co-Precursor Concentration (M)'])})

            # Define additional si co-precursor
            si_co_precursors = self.__ttcn__(row['Additional Si Co-Precursor(s)'])
            if si_co_precursors:
                si_co_precursors = str(si_co_precursors)
                si_co_precursors = si_co_precursors.split(", ")
                for i, si_co_precursor in enumerate(si_co_precursors):
                    si_co_precursor = Node("SiCoPrecursor", name=si_co_precursor)
                    self.graph.merge(si_co_precursor, "SiCoPrecursor", "name")
                    parameter_nodes[f"si_co_precursor_{i}"] = (si_co_precursor, {})  # TODO consider concentrations

            # Define surfactant
            surfactant_props = dict(
                name=self.__ttcn__(row['Surfactant'])
            )
            if surfactant_props['name']:
                surfactant = Node("Surfactant", **surfactant_props)
                self.graph.merge(surfactant, "Surfactant", "name")
                parameter_nodes['surfactant'] = (surfactant,
                                                 {"concentration": self.__ttcn__(row['Surfactant Concentration (M)'])})

            """
            This next section defines the relationships between different the different nodes
            """

            tx = self.graph.begin()

            for author in authors:
                rel_props = dict()
                tx.merge(Relationship(lit_info, "written_by", author, **rel_props))

            rel_props = dict()
            tx.merge(Relationship(final_gel, 'has', lit_info, **rel_props))

            if porosity:
                rel_props = dict(
                    porosity_percent=self.__ttcn__(row['Porosity (%)'])
                )
                tx.merge(Relationship(final_gel, 'has', porosity, **rel_props))

            if crystalline_phase:
                rel_props = dict()
                tx.merge(Relationship(final_gel, 'has', crystalline_phase, **rel_props))

            if sintering:
                rel_props = dict()
                tx.merge(Relationship(final_gel, 'sintered_by', sintering, **rel_props))

            # TODO fix this
            drying_temps = []
            for i in range(1, 5):

                drying_temp = self.__ttcn__(row[f'Drying Pressure {i} (MPa)'])
                if drying_temp:
                    drying_temps.append(drying_temp)

                if i < 3:
                    values = [self.__ttcn__(row[f'Drying Solvent']), self.__ttcn__(row[f'Drying Time {i} (hrs)']),
                              self.__ttcn__(row[f'Drying Pressure {i} (MPa)']),
                              self.__ttcn__(row[f'Drying Atmosphere {i}'])]
                    if any(values):
                        rel_props = dict(
                            step=i,
                            drying_solvent=self.__ttcn__(row[f'Drying Solvent']),
                            drying_temp=drying_temp,
                            drying_time=self.__ttcn__(row[f'Drying Time {i} (hrs)']),
                            drying_pressure=self.__ttcn__(row[f'Drying Pressure {i} (MPa)']),
                            drying_atmosphere=self.__ttcn__(row[f'Drying Atmosphere {i}'])
                        )
                    else:
                        rel_props = {}
                else:
                    values = [self.__ttcn__(row[f'Drying Solvent']), self.__ttcn__(row[f'Drying Time {i} (hrs)']),
                              self.__ttcn__(row[f'Drying Pressure {i} (MPa)'])]
                    if any(values):
                        rel_props = dict(
                            step=i,
                            drying_solvent=self.__ttcn__(row[f'Drying Solvent']),
                            drying_temp=drying_temp,
                            drying_time=self.__ttcn__(row[f'Drying Time {i} (hrs)']),
                            drying_pressure=self.__ttcn__(row[f'Drying Pressure {i} (MPa)'])
                        )
                    else:
                        rel_props = {}
                rel_matcher = RelationshipMatcher(self.graph)
                step_exists = False

                # TODO this just broke
                matches = rel_matcher.match(nodes=(final_gel, drying_method))
                if matches.__dict__['_r_type']:
                    for rel in matches.all():
                        if i == dict(rel)['step']:
                            step_exists = True
                if not step_exists:
                    tx.create(Relationship(final_gel, 'dried_by', drying_method, **rel_props))

            if drying_temps:
                rel_props = dict(
                    min_heat_treatment_temp=min(drying_temps),
                    max_heat_treatment_temp=max(drying_temps)
                )
            else:
                rel_props = dict()
            tx.merge(Relationship(non_dried_gel, 'dried_into', final_gel, **rel_props))

            if wash_solvent_1:
                rel_props = dict(
                    step=1,
                    times_washed=self.__ttcn__(row['Wash Times 1 (#)']),
                    duration=self.__ttcn__(row['Wash Duration 1 (days)']),
                    temp=self.__ttcn__(row['Wash Temp 1 (°C)'])
                )
                tx.merge(Relationship(non_dried_gel, 'washed_by', wash_solvent_1, **rel_props))

            if wash_solvent_2:
                rel_props = dict(
                    step=2,
                    times_washed=self.__ttcn__(row['Wash Times 2 (#)']),
                    duration=self.__ttcn__(row['Wash Duration 2 (days)']),
                    temp=self.__ttcn__(row['Wash Temp 2 (°C)'])
                )
                tx.merge(Relationship(non_dried_gel, 'washed_by', wash_solvent_2, **rel_props))

            if wash_solvent_3:
                rel_props = dict(
                    step=3,
                    times_washed=self.__ttcn__(row['Wash Times 3 (#)']),
                    duration=self.__ttcn__(row['Wash Duration 3 (days)']),
                    temp=self.__ttcn__(row['Wash Temp 3 (°C)'])
                )
                tx.merge(Relationship(non_dried_gel, 'washed_by', wash_solvent_3, **rel_props))

            if wash_solvent_4:
                rel_props = dict(
                    step=4,
                    times_washed=self.__ttcn__(row['Wash Times 4 (#)']),
                    duration=self.__ttcn__(row['Wash Duration 4 (days)']),
                    temp=self.__ttcn__(row['Wash Temp 4 (°C)'])
                )
                tx.merge(Relationship(non_dried_gel, 'washed_by', wash_solvent_4, **rel_props))

            tx.merge(Relationship(non_dried_gel, 'formation_by', formation_method))

            # TODO add logic to connect sol to other nodes
            if sol_1:
                tx.merge(Relationship(sol_1, 'mixed_into', non_dried_gel))

            if sol_2:
                tx.merge(Relationship(sol_2, 'mixed_into', non_dried_gel))

            if modifier_solvent and modifier:
                rel_props = dict(
                    concentration=self.__ttcn__(row['Modifier Solvent (M)'])
                )
                tx.merge(Relationship(modifier, 'uses_solvent', modifier_solvent, **rel_props))

            # Non-dried gel and final gel to parameter nodes
            for key, node_tuple in parameter_nodes.items():
                tx.merge(Relationship(final_gel, 'uses', node_tuple[0], **node_tuple[1]))
                tx.merge(Relationship(non_dried_gel, 'uses', node_tuple[0], **node_tuple[1]))

            tx.commit()


if __name__ == "__main__":
    port = 'bolt://localhost:7687'
    username = 'Neo4j'
    password = 'password'
    AerogelsToNeo4j(port, username, password, "backends/si_aerogels.csv")
