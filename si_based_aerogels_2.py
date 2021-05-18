import pandas as pd
from tqdm import tqdm

from backends.neo4j_backends import PseudoNode, PseudoRelationship, Gather
from backends.cleanup import cleanup_dataframe


def main():
    rows = cleanup_dataframe(pd.read_csv('files/si_aerogels.csv'))

    for row in tqdm(rows, total=len(rows), desc="Inserting data into Neo4j"):
        nodes: list[PseudoNode] = []
        relationships: list[PseudoRelationship] = []

        """
        These are the non-parameter nodes, or defined as core nodes. The nodes that are required in this
        section are:
            - final_gel
            - non_dired_gel
            - drying_method
            - lit_info
        """

        # TODO add logic to determine FinalGel type
        final_gel = PseudoNode("FinalGel",
                               merge_props={'index': row['Index']},
                               general_props={'final_material': row['Final Material'],
                                              'notes': row['Notes'],
                                              'pore_volume': row['Pore Volume (cm3/g)'],
                                              'average_pore_diameter': row['Average Pore Diameter (nm)'],
                                              'nanoparticle_size': row['Nanoparticle Size (nm)'],
                                              'surface_area': row['Surface Area (m2/g)'],
                                              'bulk_density': row['Bulk Density (g/cm3)'],
                                              'thermal_conductivity': row['Thermal Conductivity (W/mK)'],
                                              'young_modulus': row['Young Modulus (MPa)'],
                                              'drying_notes': row['Drying Notes']},
                               unique_prop_keys=['index'])
        nodes.append(final_gel)

        non_dried_gel = PseudoNode("NonDriedGel",
                                   merge_props={'index': row['Index']},
                                   general_props={'final_material': row['Final Material'],
                                                  'anna_notes': row["Anna's Notes"],
                                                  'final_sol_pH': row['pH final sol'],
                                                  'gelation_temp': row['Gelation Temp (°C)'],
                                                  'gelation_pressure': row['Gelation Pressure (MPa)'],
                                                  'gelation_time': row['Gelation Time (mins)'],
                                                  'hydrolysis_time': row['Hydrolysis Time (hr)'],
                                                  'aging_condition': row['Aging Conditions'],
                                                  'aging_temp': row['Aging Temp (°C)'],
                                                  'aging_time': row['Aging Time (hrs)'],
                                                  'gelation_washing_notes': row['Gelation/Washing Notes']},
                                   unique_prop_keys=['index'])
        nodes.append(non_dried_gel)

        drying_method = PseudoNode("DryingMethod",
                                   merge_props={'method': row['Drying Method']},
                                   unique_prop_keys=['method'])
        nodes.append(drying_method)

        lit_info = PseudoNode("LitInfo",
                              merge_props={'title': row['Title']},
                              general_props={'year': row['Year'],
                                             'cited_references': row['Cited References (#)'],
                                             'times_cited': row['Times Cited (#)']},
                              unique_prop_keys=['title'])
        nodes.append(lit_info)

        formation_method = PseudoNode("FormationMethod",
                                      merge_props={'method': 'Formation Method'},
                                      unique_prop_keys=['method'])
        nodes.append(formation_method)

        def __parse_authors__(raw_authors: str, corresponding: bool):
            parsed_authors = []
            if raw_authors:
                raw_authors = str(raw_authors).split(", ")
                for raw_author in raw_authors:
                    rap = raw_author.split("(")
                    if len(rap) == 2:
                        parsed_author = rap[0]
                        email = rap[1][:-1]
                    else:
                        parsed_author = raw_author
                        email = None
                    author_props = dict(
                        email=email,
                        corresponding_author=corresponding
                    )
                    parsed_author = PseudoNode("Author",
                                               merge_props={'name': parsed_author},
                                               general_props=author_props,
                                               unique_prop_keys=['name'])
                    parsed_authors.append(parsed_author)
            return parsed_authors

        authors = []
        non_corresponding_authors = __parse_authors__(row['Authors'], corresponding=False)
        corresponding_authors = __parse_authors__(row['Corresponding Author'], corresponding=True)
        authors.extend(non_corresponding_authors)
        authors.extend(corresponding_authors)
        nodes.extend(authors)

        crystalline_phase = None
        if row['Crystalline Phase']:
            crystalline_phase = PseudoNode("CrystallinePhase",
                                           merge_props={'name': row['Crystalline Phase']},
                                           unique_prop_keys=['name'])
            nodes.append(crystalline_phase)

        porosity = None
        if row['Porosity']:
            porosity = PseudoNode("Porosity",
                                  merge_props={'name': row['Porosity']},
                                  unique_prop_keys=['name'])
            nodes.append(porosity)

        wash_solvent_1 = None
        if row['Wash Solvent 1']:
            wash_solvent_1 = PseudoNode("Solvent",
                                        merge_props={'name': row['Wash Solvent 1']},
                                        unique_prop_keys=['name'])
            nodes.append(wash_solvent_1)

        wash_solvent_2 = None
        if row['Wash Solvent 2']:
            wash_solvent_2 = PseudoNode("Solvent",
                                        merge_props={'name': row['Wash Solvent 2']},
                                        unique_prop_keys=['name'])
            nodes.append(wash_solvent_2)

        wash_solvent_3 = None
        if row['Wash Solvent 3']:
            wash_solvent_3 = PseudoNode("Solvent",
                                        merge_props={'name': row['Wash Solvent 3']},
                                        unique_prop_keys=['name'])
            nodes.append(wash_solvent_3)

        wash_solvent_4 = None
        if row['Wash Solvent 4']:
            wash_solvent_4 = PseudoNode("Solvent",
                                        merge_props={'name': row['Wash Solvent 4']},
                                        unique_prop_keys=['name'])
            nodes.append(wash_solvent_4)

        modifier = None
        if row['Modifier']:
            modifier = PseudoNode("Modifier",
                                  merge_props={'name': row['Modifier']},
                                  unique_prop_keys=['name'])
            nodes.append(modifier)

        modifier_solvent = None
        if row['Modifier Solvent']:
            modifier_solvent = PseudoNode("Solvent",
                                          merge_props={'name': row['Modifier Solvent']},
                                          unique_prop_keys=['name'])
            nodes.append(modifier_solvent)

        sintering = None
        sintering_props = dict(
            notes=row['Sintering Notes'],
            atmosphere=row['Sintering Atmosphere']
        )
        if any(sintering_props):
            sintering = PseudoNode("Sintering",
                                   merge_props={'index': row['Index']},
                                   general_props=dict(final_material=row['Final Material'],
                                                      **sintering_props),
                                   unique_prop_keys=['index'])
            nodes.append(sintering)

        sol_1 = None
        sol_1_props = dict(
            compontents=row['Sol 1'],
            stir_rate=['Sol 1 Stir rate (rpm)'],
            stir_time=row['Sol 1 Stir Time (min)'],
            temp=row['Sol 1 Temp (°C)'],
        )
        if any(sol_1_props):
            sol_1 = PseudoNode("Sol",
                               merge_props={'index': row['Index']},
                               general_props=dict(final_material=row['Final Material'],
                                                  **sol_1_props),
                               unique_prop_keys=['index'])
            nodes.append(sol_1)

        sol_2 = None
        sol_2_props = dict(
            compontents=row['Sol 2'],
            stir_rate=['Sol 2 Stir rate (rpm)'],
            stir_time=row['Sol 2 Stir Time (min)'],
            temp=row['Sol 2 Temp (°C)'],
        )
        if any(sol_2_props):
            sol_2 = PseudoNode("Sol",
                               merge_props={'index': row['Index']},
                               general_props=dict(final_material=row['Final Material'],
                                                  **sol_2_props),
                               unique_prop_keys=['index'])
            nodes.append(sol_2)

        """
        This next section defines all the parameter nodes
        """

        parameter_nodes = {}

        if row['Solvent 1']:
            solvent_1 = PseudoNode("Solvent",
                                   merge_props={'name': row['Solvent 1']},
                                   unique_prop_keys=['name'])
            nodes.append(solvent_1)
            parameter_nodes['solvent_1'] = (solvent_1, {'concentration': row['Solvent 1 Concentration (M)']})

        if row['Solvent 2']:
            solvent_2 = PseudoNode("Solvent",
                                   merge_props={'name': row['Solvent 2']},
                                   unique_prop_keys=['name'])
            nodes.append(solvent_2)
            parameter_nodes['solvent_1'] = (solvent_2, {'concentration': row['Solvent 2 Concentration (M)']})

        if row['Gelation Agent']:
            gelation_agent = PseudoNode("GelationAgent",
                                        merge_props={'name': row['Gelation Agent']},
                                        unique_prop_keys=['name'])
            nodes.append(gelation_agent)
            parameter_nodes['gelation_agent'] = (gelation_agent, {'concentration': row['Gelation Agent (M)'],
                                                                  "temp": row['Gelation Temp (°C)']})

        if row['Dopant']:
            dopant = PseudoNode("Dopant",
                                merge_props={'name': row['Dopant']},
                                unique_prop_keys=['name'])
            nodes.append(dopant)
            parameter_nodes['dopant'] = (dopant, {'concentration': row['Dopant Concentration (M)']})

        if row['Acid Catalyst']:
            acid_catalyst = PseudoNode("AcidCatalyst",
                                       merge_props={'name': row['Acid Catalyst']},
                                       unique_prop_keys=['name'])
            nodes.append(acid_catalyst)
            parameter_nodes['acid_catalyst'] = (acid_catalyst,
                                                {'initial concentration': row[
                                                    'Acid Catalyst Initial Concentration (M)'],
                                                 'concentration in sol': row['Acid Catalyst concentration in Sol(M)']})

        if row['Base Catalyst']:
            base_catalyst = PseudoNode("BaseCatalyst",
                                       merge_props={'name': row['Base Catalyst']},
                                       unique_prop_keys=['name'])
            nodes.append(base_catalyst)
            parameter_nodes['base_catalyst'] = (base_catalyst,
                                                {'initial concentration': row['Base Catalyst Initial Concentration(M)'],
                                                 'concentration in sol': row['Base Catalyst concentration in Sol (M)']})

        if row['Si Precursor']:
            si_precursor = PseudoNode("SiPrecursor",
                                      merge_props={'name': row['Si Precursor']},
                                      unique_prop_keys=['name'])
            nodes.append(si_precursor)
            parameter_nodes['si_precursor'] = (si_precursor, {'concentration': row['Si Precursor Concentration (M)']})

        if row['Hybrid Aerogel Co-Precursor']:
            hybrid_precursor = PseudoNode("HybridPrecursor",
                                          merge_props={'name': row['Hybrid Aerogel Co-Precursor']},
                                          unique_prop_keys=['name'])
            nodes.append(hybrid_precursor)
            parameter_nodes['hybrid_precursor'] = (hybrid_precursor,
                                                   {'concentration': row['Co-Precursor Concentration (M)']})

        if row['Additional Si Co-Precursor(s)']:
            si_co_precursors = PseudoNode("SiCoPrecursor",
                                          merge_props={'name': row['Additional Si Co-Precursor(s)']},
                                          unique_prop_keys=['name'])
            nodes.append(si_co_precursors)
            parameter_nodes['si_co_precursors'] = (si_co_precursors, row['Si Co-Precursor Concentration (M)'])

        if row['Surfactant']:
            surfactant = PseudoNode("Surfactant",
                                    merge_props={'name': row['Surfactant']},
                                    unique_prop_keys=['name'])
            nodes.append(surfactant)
            parameter_nodes['surfactant'] = (surfactant, {'concentration': row['Surfactant Concentration (M)']})

        """
        This section defines all the relationships
        """

        for author in authors:
            lit_info_author = PseudoRelationship("written_by", lit_info, "->", author)
            relationships.append(lit_info_author)

        final_gel_lit_info = PseudoRelationship("has", final_gel, "->", lit_info)
        relationships.append(final_gel_lit_info)

        final_gel_formation_method = PseudoRelationship("formation_by", final_gel, '->', formation_method)
        relationships.append(final_gel_formation_method)

        drying_temps = []
        for i in range(1, 5):

            if i < 3:
                rel_props = dict(
                    drying_temp=row[f'Drying Temp {i} (°C)'],
                    drying_solvent=row[f'Drying Solvent'],
                    drying_time=row[f'Drying Time {i} (hrs)'],
                    drying_pressure=row[f'Drying Pressure {i} (MPa)'],
                    drying_atmosphere=row[f'Drying Atmosphere {i}']
                )
            else:
                rel_props = dict(
                    drying_temp=row[f'Drying Temp {i} (°C)'],
                    drying_solvent=row[f'Drying Solvent'],
                    drying_time=row[f'Drying Time {i} (hrs)'],
                    drying_pressure=row[f'Drying Pressure {i} (MPa)'],
                )
            if any(rel_props):
                rel_props['step'] = i
                final_gel_drying_method = PseudoRelationship("dried_by", final_gel, "->", drying_method,
                                                             merge_props={'step': i})
                relationships.append(final_gel_drying_method)
                if isinstance(rel_props['drying_temp'], float) or isinstance(rel_props['drying_temp'], int):
                    drying_temps.append(rel_props['drying_temp'])

        if drying_temps:
            non_dried_final_gel = PseudoRelationship("dried_into", non_dried_gel, "->", final_gel,
                                                     general_props={'min_heat_treatment_temp': min(drying_temps),
                                                                    'max_heat_treatment_temp': max(drying_temps)})
        else:
            non_dried_final_gel = PseudoRelationship("dried_into", non_dried_gel, "->", final_gel)
        relationships.append(non_dried_final_gel)

        if porosity:
            final_gel_porosity = PseudoRelationship("has", final_gel, '->', porosity,
                                                    general_props={' porosity_percent': row['Porosity (%)']})
            relationships.append(final_gel_porosity)

        if crystalline_phase:
            final_gel_crystalline = PseudoRelationship("has", final_gel, '->', crystalline_phase)
            relationships.append(final_gel_crystalline)

        if sintering:
            final_gel_sintering = PseudoRelationship("sintered_by", final_gel, '->', sintering)
            relationships.append(final_gel_sintering)

        if wash_solvent_1:
            non_dried_washed_1 = PseudoRelationship("washed_by", non_dried_gel, "->", wash_solvent_1,
                                                    merge_props={"step": 1},
                                                    general_props={"times_washed": row['Wash Times 1 (#)'],
                                                                   "duration": row['Wash Duration 1 (days)'],
                                                                   "temp": row['Wash Temp 1 (°C)']})
            relationships.append(non_dried_washed_1)

        if wash_solvent_2:
            non_dried_washed_2 = PseudoRelationship("washed_by", non_dried_gel, "->", wash_solvent_2,
                                                    merge_props={"step": 2},
                                                    general_props={"times_washed": row['Wash Times 2 (#)'],
                                                                   "duration": row['Wash Duration 2 (days)'],
                                                                   "temp": row['Wash Temp 2 (°C)']})
            relationships.append(non_dried_washed_2)

        if wash_solvent_3:
            non_dried_washed_3 = PseudoRelationship("washed_by", non_dried_gel, "->", wash_solvent_3,
                                                    merge_props={"step": 3},
                                                    general_props={"times_washed": row['Wash Times 3 (#)'],
                                                                   "duration": row['Wash Duration 3 (days)'],
                                                                   "temp": row['Wash Temp 3 (°C)']})
            relationships.append(non_dried_washed_3)

        if wash_solvent_4:
            non_dried_washed_4 = PseudoRelationship("washed_by", non_dried_gel, "->", wash_solvent_4,
                                                    merge_props={"step": 4},
                                                    general_props={"times_washed": row['Wash Times 4 (#)'],
                                                                   "duration": row['Wash Duration 4 (days)'],
                                                                   "temp": row['Wash Temp 4 (°C)']})
            relationships.append(non_dried_washed_4)

        if sol_1:
            sol_1_non_dired = PseudoRelationship("mixed_into", sol_1, '->', non_dried_gel)
            relationships.append(sol_1_non_dired)

        if sol_2:
            sol_2_non_dried = PseudoRelationship("mixed_into", sol_2, '->', non_dried_gel)
            relationships.append(sol_2_non_dried)

        if modifier:
            final_gel_modifier = PseudoRelationship("uses", final_gel, "->", modifier)
            non_dried_modifier = PseudoRelationship("uses", non_dried_gel, '->', modifier)
            relationships.append(final_gel_modifier)
            relationships.append(non_dried_modifier)

        if modifier_solvent and modifier:
            modifier_modifier_solvent = PseudoRelationship("uses_solvent", modifier, '->', modifier_solvent,
                                                           general_props={"concentration": row['Modifier Solvent (M)']})
            relationships.append(modifier_modifier_solvent)

        for key, node_tuple in parameter_nodes.items():
            final_gel_parameter = PseudoRelationship("uses", final_gel, "->", node_tuple[0],
                                                     general_props=node_tuple[1])
            non_dried_parameter = PseudoRelationship("uses", non_dried_gel, "->", node_tuple[0],
                                                     general_props=node_tuple[1])
            relationships.append(final_gel_parameter)
            relationships.append(non_dried_parameter)

        gathered = Gather(nodes, relationships)
        gathered.merge()


if __name__ == '__main__':
    main()
