from ast import literal_eval

import pandas as pd
from tqdm import tqdm
from py2neo import Graph


molarity_data = pd.read_excel('backends/dev_export.xlsx')


def __ttcn__(n):
    if not n:
        return n
    if str(n) == "nan":
        return None
    try:
        return float(n)
    except ValueError:
        return n


def __fmoc__(final_product, n):  # Find molarity of compound
    matched_row = molarity_data.loc[molarity_data['Final Material'] == final_product].to_dict('records')
    if matched_row:
        matched_row = matched_row[0]
        molarities = matched_row['calculated_molarities']
        if str(molarities) != "nan":
            molarities = literal_eval(molarities)
            if n in molarities.keys():
                return molarities[n]
    return None


def core_node(row):
    gel_id = row['Index']

    final_product = row['Final Material']
    core_notes = row['Notes']
    pore_volume = __ttcn__(row['Pore Volume (cm3/g)'])
    average_pore_diameter = __ttcn__(row['Average Pore Diameter (nm)'])
    nano_particle_size = __ttcn__(row['Nanoparticle Size (nm)'])
    surface_area = __ttcn__(row['Surface Area (m2/g)'])
    bulk_density = __ttcn__(row['Bulk Density (g/cm3)'])
    thermal_conductivity = __ttcn__(row['Thermal Conductivity (W/mK)'])
    heat_treatment_temp = __ttcn__(row['Sintering Temp (°C)'])
    young_modulus = __ttcn__(row['Young Modulus (MPa)'])

    graph.evaluate(

        """
        MERGE (a: Aerogel {id: $id})
            ON CREATE SET a.final_product = $final_product, a.core_notes = $core_notes, a.surface_area = $surface_area,
                a.heat_treatment_temp = $heat_treatment_temp, a.young_modulus = $young_modulus

        MERGE (s: Synthesis {id: $id})
            ON CREATE SET s.name = "Synthesis"

        MERGE (s)-[n:synthesizes]->(a)
            ON CREATE SET n.pore_volume = $pore_volume, n.average_pore_diameter = $average_pore_diameter, 
                n.nano_particle_size = $nano_particle_size, n.surface_area = $surface_area,
                n.bulk_density = $bulk_density, n.thermal_conductivity = $thermal_conductivity

        """, parameters={"id": gel_id, "final_product": final_product, "pore_volume": pore_volume,
                         "average_pore_diameter": average_pore_diameter, "nano_particle_size": nano_particle_size,
                         "surface_area": surface_area, "bulk_density": bulk_density, "thermal_conductivity": thermal_conductivity,
                         "core_notes": core_notes, "heat_treatment_temp": heat_treatment_temp,
                         "young_modulus": young_modulus}
    )

    porosity = row['Porosity']
    porosity_percent = __ttcn__(row['Porosity (%)'])
    if porosity is not None:
        porosities = porosity.split(', ')
        for porosity in porosities:
            graph.evaluate(

                """
                MATCH (n: Aerogel {id: $id})
                MERGE (p: Porosity {name: $porosity})
                MERGE (n)-[rel:has_porosity]->(p)
                    SET rel.porosity_percent = $porosity_percent
                """, parameters={"id": gel_id, "porosity": porosity,
                                 "porosity_percent": porosity_percent}

            )

    crystalline_phase = row['Crystalline Phase']
    if crystalline_phase is not None:
        crystalline_phases = crystalline_phase.split(', ')
        for crystalline_phase in crystalline_phases:
            graph.evaluate(

                """
                MATCH (n: Aerogel {id: $id})
                MERGE (c: CrystallinePhase {name: $crystalline_phase})
                MERGE (n)-[:has_crystalline_phase]->(c)
                """, parameters={"id": gel_id, "crystalline_phase": crystalline_phase}

            )


def sintering(row):
    gel_id = row['Index']
    sintering_temp = __ttcn__(row['Sintering Temp (°C)'])
    sintering_time = __ttcn__(row['Sintering Time (min)'])
    sintering_ramp_rate = __ttcn__(row['Ramp Rate (°C/min)'])
    sintering_notes = row['Sintering Notes']

    graph.evaluate(

        """
        MATCH (a: Aerogel {id: $id})

        MERGE (s:Sintering {id: $id})
            ON CREATE SET s.sintering_notes = $sintering_notes

        MERGE (a)-[n:uses_sintering]->(s)
            ON CREATE SET n.sintering_temp = $sintering_temp, n.sintering_time = $sintering_time,
                n.sintering_ramp_rate = $sintering_ramp_rate

        """, parameters={"id": gel_id, "sintering_temp": sintering_temp, "sintering_time": sintering_time,
                         "sintering_ramp_rate": sintering_ramp_rate,
                         "sintering_notes": sintering_notes})

    sintering_atmosphere = row['Sintering Atmosphere']
    if sintering_atmosphere is not None:
        graph.evaluate(
            """
            MATCH (s:Sintering {id: $id})
            MERGE (a:SinteringAtmosphere {name: $sintering_atmosphere})
            MERGE (s)-[:has_atmosphere]->(a)
            """, parameters={"id": gel_id, "sintering_atmosphere": sintering_atmosphere}
        )


def lit_info(row):
    gel_id = row['Index']
    author = row['Author']
    title = row['Title']
    year = row['Year']
    cited_reference = row['Cited References (#)']
    times_cited = row['Times Cited (#)']

    graph.evaluate(

        """
        MATCH (a:Synthesis {id: $id})
        MERGE (lit:LitInfo {id: $title})
            ON CREATE SET lit.author = $author, lit.title = $title
        MERGE (a)-[n:has_lit_info]->(lit)
            ON CREATE SET n.year = $year, n.cited_reference = $cited_reference, n.times_cited = $times_cited

        """, parameters={"id": gel_id, "author": author, "title": title, "year": year,
                         "cited_reference": cited_reference, "times_cited": times_cited}

    )


def gel(row):
    gel_id = row['Index']
    annas_notes = row["Anna's Notes"]
    ph_sol = __ttcn__(row['pH final sol'])
    gelation_temp = __ttcn__(row['Gelation Temp (°C)'])
    gelation_pressure = __ttcn__(row['Gelation Pressure (MPa)'])
    gelation_time = __ttcn__(row['Gelation Time (mins)'])
    heat_treatment_temp = __ttcn__(row['Sintering Temp (°C)'])
    surface_area = __ttcn__(row['Surface Area (m2/g)'])
    hydrolysis_time = __ttcn__(row['Hydrolysis Time (hr)'])

    aging_conditions_1 = row['Aging Conditions']
    aging_temp_1 = __ttcn__(row['Aging Temp (°C)'])
    aging_time_1 = __ttcn__(row['Aging Time (hrs)'])

    aging_conditions_2 = row['Aging Conditions 2']
    aging_temp_2 = __ttcn__(row['Aging Temp 2 (°C)'])
    aging_time_2 = __ttcn__(row['Aging Time 2 (hrs)'])

    graph.evaluate(

        """
        MATCH (a:Synthesis {id: $id})

        MERGE (m:Gel {id: $id})
            ON CREATE SET m.name = "Gel", m.surface_area = $surface_area, m.heat_treatment_temp = $heat_treatment_temp

        MERGE (a)-[g:uses_gel]->(m)
            SET g.annas_notes = $annas_notes, g.ph_sol = $ph_sol, g.gelation_temp = $gelation_temp,
                g.gelation_pressure = $gelation_pressure, g.gelation_time = $gelation_time,
                g.aging_time_1 = $aging_time_1, g.aging_temp_1 = $aging_temp_1, 
                g.aging_conditions_1 = $aging_conditions_1, g.hydrolysis_time = $hydrolysis_time,
                g.aging_conditions_2 = $aging_conditions_2, g.aging_temp_2 = $aging_temp_2,
                g.aging_time_2 = $aging_time_2
                

        """, parameters={"id": gel_id, "annas_notes": annas_notes,
                         "ph_sol": ph_sol, "gelation_temp": gelation_temp, "gelation_pressure": gelation_pressure,
                         "gelation_time": gelation_time, "aging_time_1": aging_time_1, "aging_temp_1": aging_temp_1,
                         "aging_conditions_1": aging_conditions_1, "hydrolysis_time": hydrolysis_time,
                         "surface_area": surface_area, "heat_treatment_temp": heat_treatment_temp,
                         "aging_conditions_2": aging_conditions_2, "aging_temp_2": aging_temp_2,
                         "aging_time_2": aging_time_2}

    )

    formation_method = row['Formation Method']
    if formation_method is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (f:FormationMethod {name: $formation_method})
            MERGE (g)-[:uses_formation_method]->(f)

            """, parameters={"id": gel_id, "formation_method": formation_method}

        )

    si_precusor = row['Si Precursor']
    si_precusor_conc = __ttcn__(row['Si Precursor Concentration (M)'])
    if si_precusor is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:SiPrecusor {name: $si_precusor})
            MERGE (g)-[rel:uses_si_precusor]->(s)
                SET rel.si_precusor_conc = $si_precusor_conc

            """, parameters={"id": gel_id, "si_precusor": si_precusor, "si_precusor_conc": si_precusor_conc}

        )

    additional_si_co_precursors = row['Additional Si Co-Precursor(s)']
    additional_si_co_precursor_conc = __ttcn__(row['Si Co-Precursor Concentration (M)'])
    if additional_si_co_precursors is not None:
        additional_si_co_precursors = additional_si_co_precursors.split(', ')
        for additional_si_co_precursor in additional_si_co_precursors:
            graph.evaluate(

                """
                MATCH (g:Gel {id: $id})

                MERGE (s:SiCoPrecursor {name: $additional_si_co_precursor})
                MERGE (g)-[r:has_additional_si_co_precursor]->(s)
                    SET r.concentration = $additional_si_co_precursor_conc

                """, parameters={"id": gel_id, "additional_si_co_precursor": additional_si_co_precursor,
                                 "additional_si_co_precursor_conc": additional_si_co_precursor_conc}

            )

    hybird_co_precursor = row['Hybrid Aerogel Co-Precursor']
    hybird_co_precursor_conc = __ttcn__(row['Co-Precursor Concentration (M)'])
    if hybird_co_precursor is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:HybirdPrecursor {name: $hybird_co_precursor})
            MERGE (g)-[rel:uses_hybird_co_precursor]->(s)
                SET rel.concentration = $hybird_co_precursor_conc

            """, parameters={"id": gel_id, "hybird_co_precursor": hybird_co_precursor, "hybird_co_precursor_conc": hybird_co_precursor_conc}

        )

    acid_catalyst = row['Acid Catalyst']
    acid_catalyst_conc = __ttcn__(row['Acid Catalyst Initial Concentration (M)'])
    if acid_catalyst is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:AcidCatalyst {name: $acid_catalyst})
            MERGE (g)-[rel:uses_acid_catalyst]->(s)
                SET rel.concentration = $acid_catalyst_conc

            """, parameters={"id": gel_id, "acid_catalyst": acid_catalyst, "acid_catalyst_conc": acid_catalyst_conc}

        )

    base_catalyst = row['Base Catalyst']
    base_catalyst_conc = __ttcn__(row['Base Catalyst Initial Concentration(M)'])
    if base_catalyst is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:BaseCatalyst {name: $base_catalyst})
            MERGE (g)-[rel:uses_base_catalyst]->(s)
                SET rel.concentration = $base_catalyst_conc

            """, parameters={"id": gel_id, "base_catalyst": base_catalyst, "base_catalyst_conc": base_catalyst_conc}

        )

    final_product = row['Final Material']

    sol_1_stir_rate = __ttcn__(row['Sol 1 Stir rate (rpm)'])
    sol_1_stir_time = __ttcn__(row['Sol 1 Stir Time (min)'])
    sol_1_standing_time = __ttcn__(row['Sol 1 Standing time (hr)'])
    sol_1_temp = __ttcn__(row['Sol 1 Temp (°C)'])
    sol_1_ph = __ttcn__(row['pH Sol 1'])

    sol_1_props = [sol_1_stir_rate, sol_1_stir_time, sol_1_standing_time, sol_1_temp, sol_1_ph]
    if any(sol_1_props):
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:Sol {id: $id, solNumber: 1})
            ON CREATE SET s.stir_rate = $sol_1_stir_rate, s.stir_time = $sol_1_stir_time, 
                s.standing_time = $sol_1_standing_time, s.temp = $sol_1_temp, s.ph = $sol_1_ph

            """, parameters={"id": gel_id, "sol_1_stir_rate": sol_1_stir_rate,
                             "sol_1_stir_time": sol_1_stir_time, "sol_1_standing_time": sol_1_standing_time,
                             "sol_1_temp": sol_1_temp, "sol_1_ph": sol_1_ph}

        )

    sol_1_compounds = row['Sol 1']
    if sol_1_compounds is not None:
        for sol_1_compound in sol_1_compounds.split(', '):
            molarity = __fmoc__(final_product, sol_1_compound)
            graph.evaluate(

                """
                MATCH (g:Gel {id: $id})
                MATCH (sol:Sol {id: $id, solNumber: 1})
                MERGE (g)-[:uses_sol]->(sol)

                MERGE (sov:SolSolvent {name: $sol_1_compound})
                MERGE (sol)-[rel:has_solvent]->(sov)
                    SET rel.molarity = $molarity

                """, parameters={"id": gel_id, "sol_1_compound": sol_1_compound, "molarity": molarity}

            )

    sol_2_stir_rate = __ttcn__(row['Sol 2 Stir rate (rpm)'])
    sol_2_stir_time = __ttcn__(row['Sol 2 Stir Time (min)'])
    sol_2_standing_time = __ttcn__(row['Sol 2 Standing time (hr)'])
    sol_2_temp = __ttcn__(row['Sol 2 Temp (°C)'])
    sol_2_ph = __ttcn__(row['pH Sol 2'])

    sol_2_props = [sol_2_stir_rate, sol_2_stir_time, sol_2_standing_time, sol_2_temp, sol_2_ph]
    if any(sol_2_props):
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:Sol {id: $id, solNumber: 1})
            ON CREATE SET s.stir_rate = $sol_2_stir_rate, s.stir_time = $sol_2_stir_time, 
                s.standing_time = $sol_2_standing_time, s.temp = $sol_2_temp, s.ph = $sol_2_ph

            """, parameters={"id": gel_id, "sol_2_stir_rate": sol_2_stir_rate,
                             "sol_2_stir_time": sol_2_stir_time, "sol_2_standing_time": sol_2_standing_time,
                             "sol_2_temp": sol_2_temp, "sol_2_ph": sol_2_ph}

        )

    sol_2_compounds = row['Sol 2']
    if sol_2_compounds is not None:
        for sol_2_compound in sol_2_compounds.split(', '):
            molarity = __fmoc__(final_product, sol_2_compound)
            graph.evaluate(

                """
                MATCH (g:Gel {id: $id})
                MATCH (sol:Sol {id: $id, solNumber: 1})
                MERGE (g)-[:uses_sol]->(sol)

                MERGE (sov:SolSolvent {name: $sol_2_compound})
                MERGE (sol)-[rel:has_solvent]->(sov)
                    SET rel.molarity = $molarity

                """, parameters={"id": gel_id, "sol_2_compound": sol_2_compound, "molarity": molarity}

            )

    dopants = row['Dopant']
    dopant_conc = __ttcn__(row['Dopant Concentration (M)'])
    if dopants is not None:
        dopants = dopants.split(', ')

        if len(dopants) == 1:
            graph.evaluate(

                """
                MATCH (g:Gel {id: $id})
                    SET g.total_dopant_conc = $dopant_conc
                MERGE (d:Dopant {name: $dopant})
                MERGE (g)-[rel:uses_dopant]->(d)
                    SET rel.dopant_conc = $dopant_conc

                """, parameters={"id": gel_id, "dopant": dopants[0], "dopant_conc": dopant_conc}

            )

        else:
            for dopant in dopants:
                graph.evaluate(

                    """
                    MATCH (g:Gel {id: $id})
                        SET g.total_dopant_conc = $dopant_conc
                    MERGE (d:Dopant {name: $dopant})
                    MERGE (g)-[rel:uses_dopant]->(d)
                    """, parameters={"id": gel_id, "dopant": dopant, "dopant_conc": dopant_conc}

                )

    gel_solvent_1 = row['Solvent 1']
    gel_solvent_1_conc = __ttcn__(row['Solvent 1 Concentration (M)'])
    if gel_solvent_1 is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:GelSolvent {name: $gel_solvent_1})
            MERGE (g)-[rel:uses_gel_solvent]->(s)
                SET rel.gel_solvent_conc = $gel_solvent_1_conc

            """, parameters={"id": gel_id, "gel_solvent_1": gel_solvent_1, "gel_solvent_1_conc": gel_solvent_1_conc}

        )

    gel_solvent_2 = row['Solvent 2']
    gel_solvent_2_conc = __ttcn__(row['Solvent 2 Concentration (M)'])
    if gel_solvent_2 is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:GelSolvent {name: $gel_solvent_2})
            MERGE (g)-[rel:uses_gel_solvent]->(s)
                SET rel.gel_solvent_conc = $gel_solvent_2_conc

            """, parameters={"id": gel_id, "gel_solvent_2": gel_solvent_2, "gel_solvent_2_conc": gel_solvent_2_conc}

        )

    additional_solvents = row['Additional Solvents']
    if additional_solvents is not None:
        additional_solvents = additional_solvents.split(', ')
        for additional_solvent in additional_solvents:
            graph.evaluate(

                """
                MATCH (g:Gel {id: $id})

                MERGE (s:GelSolvent {name: $additional_solvent})
                MERGE (g)-[:has_additional_solvent]->(s)

                """, parameters={"id": gel_id, "additional_solvent": additional_solvent}

            )

    modifier = row['Modifier']
    modifier_conc = __ttcn__(row['Modifier Concentration (M)'])
    if modifier is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (m:Modifier {name: $modifier})
            MERGE (g)-[rel:uses_modifier]->(m)
                SET rel.modifier_conc = $modifier_conc

            """, parameters={"id": gel_id, "modifier": modifier, "modifier_conc": modifier_conc}

        )

    surfactant = row['Surfactant']
    surfactant_conc = __ttcn__(row['Surfactant Concentration (M)'])
    if surfactant is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:Surfactant {name: $surfactant})
            MERGE (g)-[rel:uses_surfactant]->(s)
                SET rel.surfactant_conc = $surfactant_conc

            """, parameters={"id": gel_id, "surfactant": surfactant, "surfactant_conc": surfactant_conc}

        )

    gelation_agents = row['Gelation Agent']
    gelation_agent_conc = __ttcn__(row['Gelation Agent (M)'])
    if gelation_agents is not None:
        gelation_agents = gelation_agents.split(', ')

        if len(gelation_agents) == 1:

            graph.evaluate(

                """
                MATCH (g:Gel {id: $id})

                MERGE (a:GelationAgent {name: $gelation_agent})
                MERGE (g)-[rel:uses_gelation_agent]->(a)
                    SET rel.gelation_agent_conc = $gelation_agent_conc

                """, parameters={"id": gel_id, "gelation_agent": gelation_agents[0],
                                 "gelation_agent_conc": gelation_agent_conc}

            )

        else:
            for gelation_agent in gelation_agents:
                graph.evaluate(

                    """
                    MATCH (g:Gel {id: $id})

                    MERGE (a:GelationAgent {name: $gelation_agent})
                    MERGE (g)-[rel:uses_gelation_agent]->(a)

                    """, parameters={"id": gel_id, "gelation_agent": gelation_agent,
                                     "gelation_agent_conc": gelation_agent_conc}

                )


def washing_steps(row):
    gel_id = row['Index']
    washing_notes = row['Gelation/Washing Notes']

    graph.evaluate(
        """
        MATCH (a:Synthesis {id: $id})
        MERGE (w:WashingSteps {id: $id})
            ON CREATE SET w.notes = $notes
        MERGE (a)-[:uses_washing_steps]->(w)
        """, parameters={"id": gel_id, "notes": washing_notes}
    )

    wash_solvent_1 = row['Wash Solvent 1']
    wash_times_1 = row['Wash Times 1 (#)']
    wash_duration_1 = __ttcn__(row['Wash Duration 1 (days)'])
    wash_temp_1 = __ttcn__(row['Wash Temp 1 (°C)'])
    if wash_solvent_1 is not None:
        graph.evaluate(
            """
            MATCH (w:WashingSteps {id: $id})

            MERGE (m:WashingStep {solvent: $wash_solvent})
            MERGE (w)-[s:washing_step]->(m)
                SET s.wash_times = $wash_times, s.wash_duration = $wash_duration,
                    s.wash_temp = $wash_temp

            """, parameters={"id": gel_id, "wash_solvent": wash_solvent_1, "wash_times": wash_times_1,
                             "wash_duration": wash_duration_1, "wash_temp": wash_temp_1}
        )

    wash_solvent_2 = row['Wash Solvent 2']
    wash_times_2 = row['Wash Times 2 (#)']
    wash_duration_2 = __ttcn__(row['Wash Duration 2 (days)'])
    wash_temp_2 = __ttcn__(row['Wash Temp 2 (°C)'])
    if wash_solvent_2 is not None:
        graph.evaluate(
            """
            MATCH (w:WashingSteps {id: $id})

            MERGE (m:WashingStep {solvent: $wash_solvent})
            MERGE (w)-[s:washing_step]->(m)
                SET s.wash_times = $wash_times, s.wash_duration = $wash_duration,
                    s.wash_temp = $wash_temp

            """, parameters={"id": gel_id, "wash_solvent": wash_solvent_2, "wash_times": wash_times_2,
                             "wash_duration": wash_duration_2, "wash_temp": wash_temp_2}
        )

    wash_solvent_3 = row['Wash Solvent 3']
    wash_times_3 = row['Wash Times 3 (#)']
    wash_duration_3 = __ttcn__(row['Wash Duration 3 (days)'])
    wash_temp_3 = __ttcn__(row['Wash Temp 3 (°C)'])
    if wash_solvent_3 is not None:
        graph.evaluate(
            """
            MATCH (w:WashingSteps {id: $id})

            MERGE (m:WashingStep {solvent: $wash_solvent})
            MERGE (w)-[s:washing_step]->(m)
                SET s.wash_times = $wash_times, s.wash_duration = $wash_duration,
                    s.wash_temp = $wash_temp

            """, parameters={"id": gel_id, "wash_solvent": wash_solvent_3, "wash_times": wash_times_3,
                             "wash_duration": wash_duration_3, "wash_temp": wash_temp_3}
        )

    wash_solvent_4 = row['Wash Solvent 4']
    wash_times_4 = row['Wash Times 4 (#)']
    wash_duration_4 = __ttcn__(row['Wash Duration 4 (days)'])
    wash_temp_4 = __ttcn__(row['Wash Temp 4 (°C)'])
    if wash_solvent_4 is not None:
        graph.evaluate(
            """
            MATCH (w:WashingSteps {id: $id})

            MERGE (m:WashingStep {solvent: $wash_solvent})
            MERGE (w)-[s:washing_step]->(m)
                SET s.wash_times = $wash_times, s.wash_duration = $wash_duration,
                    s.wash_temp = $wash_temp

            """, parameters={"id": gel_id, "wash_solvent": wash_solvent_4, "wash_times": wash_times_4,
                             "wash_duration": wash_duration_4, "wash_temp": wash_temp_4}
        )


def drying(row):
    gel_id = row['Index']
    drying_method = row['Drying Method']
    drying_solvent = row['Drying Solvent']
    drying_notes = row['Drying Notes']

    drying_temp_1 = __ttcn__(row['Drying Temp (°C)'])
    drying_pressure_1 = __ttcn__(row['Drying Pressure (MPa)'])
    drying_time_1 = __ttcn__(row['Drying Time (hrs)'])
    drying_atmosphere_1 = __ttcn__(row['Drying Atmosphere'])

    graph.evaluate(
        """
        MATCH (a:Synthesis {id: $id})
        MERGE (n:DryingStep {id: $id, step: 1})
            ON CREATE SET n.notes = $notes, n.drying_atmosphere = $drying_atmosphere
        MERGE (a)-[d:uses_drying_step]->(n)
            ON CREATE SET d.drying_temp = $drying_temp,
                    d.drying_pressure = $drying_pressure, d.drying_time = $drying_time
        """, parameters={"id": gel_id, "notes": drying_notes, "drying_temp": drying_temp_1,
                         "drying_pressure": drying_pressure_1, "drying_time": drying_time_1,
                         "drying_atmosphere": drying_atmosphere_1}
    )

    if drying_method is not None:
        graph.evaluate(
            """
            MATCH (a:DryingStep {id: $id, step: 1})
            MERGE (m:DryingMethod {drying_method: $drying_method})    
            MERGE (a)-[d:uses_drying_method]->(m)
            """, parameters={"id": gel_id, "drying_method": drying_method}
        )

    if drying_solvent is not None:
        graph.evaluate(
            """
            MATCH (a:DryingStep {id: $id, step: 1})
            MERGE (d:DryingSolvent {solvent: $drying_solvent})
            MERGE (a)-[:uses_drying_solvent]->(d)
            """, parameters={"id": gel_id, "drying_solvent": drying_solvent}
        )

    drying_temp_2 = __ttcn__(row['Drying Temp 2 (°C)'])
    drying_pressure_2 = __ttcn__(row['Drying Pressure 2 (MPa)'])
    drying_time_2 = __ttcn__(row['Drying Time 2 (hrs)'])
    drying_atmosphere_2 = __ttcn__(row['Drying Atmosphere 2'])

    twos = [drying_temp_2, drying_pressure_2, drying_time_2, drying_atmosphere_2]

    if any(twos):

        graph.evaluate(
            """
            MATCH (a:Synthesis {id: $id})
            MERGE (n:DryingStep {id: $id, step: 2})
                ON CREATE SET n.notes = $notes, n.drying_atmosphere = $drying_atmosphere
            MERGE (a)-[d:uses_drying_step]->(n)
                ON CREATE SET d.drying_temp = $drying_temp, d.drying_pressure = $drying_pressure, 
                d.drying_time = $drying_time
            """, parameters={"id": gel_id, "notes": drying_notes, "drying_temp": drying_temp_2,
                             "drying_pressure": drying_pressure_2, "drying_time": drying_time_2,
                             "drying_atmosphere": drying_atmosphere_2}
        )

        if drying_method is not None:
            graph.evaluate(
                """
                MATCH (a:DryingStep {id: $id, step: 2})
                MERGE (m:DryingMethod {drying_method: $drying_method})    
                MERGE (a)-[d:uses_drying]->(m)
                """, parameters={"id": gel_id, "drying_method": drying_method}
            )

        if drying_solvent is not None:
            graph.evaluate(
                """
                MATCH (a:DryingStep {id: $id, step: 2})
                MERGE (d:DryingSolvent {solvent: $drying_solvent})
                MERGE (a)-[:uses_drying_solvent]->(d)
                """, parameters={"id": gel_id, "drying_solvent": drying_solvent}
            )


def main():
    print("Inserting data into Neo4j...")
    df = pd.read_csv('backends/si_aerogels.csv')
    df = df.where(pd.notnull(df), None)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    rows = df.to_dict('records')
    for row in tqdm(rows):
        core_node(row)
        sintering(row)
        lit_info(row)
        gel(row)
        washing_steps(row)
        drying(row)
    print('Done')


if __name__ == "__main__":
    port = 'bolt://localhost:7687'
    username = 'Neo4j'
    password = 'password'
    graph = Graph(port, username=username, password=password)
    main()
