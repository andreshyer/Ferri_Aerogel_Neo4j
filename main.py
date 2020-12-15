import pandas as pd
from py2neo import Graph


def core_node(row):
    gel_id = row['Index']

    final_product = row['Final Material']
    pore_volume = row['Pore Volume (cm3/g)']
    pore_size = row['Pore Size (nm)']
    nano_particle_size = row['Nanoparticle Size (nm)']
    surface_area = row['Surface Area (m2/g)']
    density = row['Density (g/cm3)']
    thermal_conductivity = row['Thermal Conductivity (W/mK)']
    core_notes = row['Notes']

    graph.evaluate(

        """
        MERGE (n: Aerogel {id: $id})
            ON CREATE SET n.final_product = $final_product, n.pore_volume = $pore_volume, n.pore_size = $pore_size,
                n.nano_particle_size = $nano_particle_size, n.surface_area = $surface_area, n.density = $density,
                n.thermal_conductivity = $thermal_conductivity, n.core_notes = $core_notes
                
        MERGE (s: Synthesis {id: $id})
            ON CREATE SET s.name = "Synthesis"
        
        MERGE (s)-[:synthesizes]->(n)
        
        """, parameters={"id": gel_id, "final_product": final_product, "pore_volume": pore_volume,
                         "pore_size": pore_size, "nano_particle_size": nano_particle_size,
                         "surface_area": surface_area, "density": density, "thermal_conductivity": thermal_conductivity,
                         "core_notes": core_notes}
    )

    porosity = row['Porosity']
    porosity_percent = row['Porosity (%)']
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
    sintering_temp = row['Sintering Temp (°C)']
    sintering_time = row['Sintering Time (min)']
    sintering_ramp_rate = row['Ramp Rate (°C/min)']
    sintering_notes = row['Sintering Notes']

    graph.evaluate(

        """
        MATCH (a: Aerogel {id: $id})
        
        MERGE (n:Sintering {id: $id})
            ON CREATE SET n.sintering_temp = $sintering_temp, n.sintering_time = $sintering_time,
                n.sintering_ramp_rate = $sintering_ramp_rate, n.sintering_notes = $sintering_notes
                
        MERGE (a)-[:uses_sintering]->(n)
    
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
            ON CREATE SET lit.author = $author, lit.title = $title, lit.year = $year, 
                lit.cited_reference = $cited_reference, lit.times_cited = $times_cited
        MERGE (a)-[:has_lit_info]->(lit)
        
        """, parameters={"id": gel_id, "author": author, "title": title, "year": year,
                         "cited_reference": cited_reference, "times_cited": times_cited}

    )


def gel(row):
    gel_id = row['Index']
    annas_notes = row["Anna's Notes"]
    ph_sol = row['pH final sol']
    gelation_temp = row['Gelation Temp (°C)']
    gelation_pressure = row['Gelation Pressure (MPa)']
    gelation_time = row['Gelation Time (mins)']
    aging_temp = row['Aging Temp (°C)']
    aging_time = row['Aging Time (hrs)']

    graph.evaluate(

        """
        MATCH (a:Synthesis {id: $id})
        
        MERGE (m:Gel {id: $id})
            ON CREATE SET m.name = "Gel"
            
        MERGE (a)-[g:uses_gel]->(m)
            SET g.annas_notes = $annas_notes, g.ph_sol = $ph_sol, g.gelation_temp = $gelation_temp,
                g.gelation_pressure = $gelation_pressure, g.gelation_time = $gelation_time,
                g.aging_time = $aging_time, g.aging_temp = $aging_temp
        
        """, parameters={"id": gel_id, "annas_notes": annas_notes,
                         "ph_sol": ph_sol, "gelation_temp": gelation_temp, "gelation_pressure": gelation_pressure,
                         "gelation_time": gelation_time, "aging_time": aging_time, "aging_temp": aging_temp}

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

    zr_precusor = row['Zr Precursor']
    zr_precusor_conc = row['Zr Precursor Concentration (M)']
    if zr_precusor is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (z:ZrPrecusor {name: $zr_precusor})
            MERGE (g)-[rel:uses_zr_precusor]->(z)
                SET rel.zr_precusor_conc = $zr_precusor_conc

            """, parameters={"id": gel_id, "zr_precusor": zr_precusor, "zr_precusor_conc": zr_precusor_conc}

        )

    dopant = row['Dopant']
    dopant_conc = row['Dopant Concentration (M)']
    if dopant is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (d:Dopant {name: $dopant})
            MERGE (g)-[rel:uses_dopant]->(d)
                SET rel.dopant_conc = $dopant_conc

            """, parameters={"id": gel_id, "dopant": dopant, "dopant_conc": dopant_conc}

        )

    gel_solvent_1 = row['Solvent 1']
    gel_solvent_1_conc = row['Solvent 1 Concentration (M)']
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
    if gel_solvent_2 is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:GelSolvent {name: $gel_solvent_2})
            MERGE (g)-[:uses_gel_solvent]->(s)

            """, parameters={"id": gel_id, "gel_solvent_2": gel_solvent_2}

        )

    additional_solvents_precusors = row['Additional Solvents/DCCA/Precursors (non Zr)']
    if additional_solvents_precusors is not None:
        additional_solvents_precusors = additional_solvents_precusors.split(', ')
        for additional_solvents_precusor in additional_solvents_precusors:
            graph.evaluate(

                """
                MATCH (g:Gel {id: $id})
    
                MERGE (s:AdditionalSolventsAndPrecusors {name: $additional_solvents_precusors})
                MERGE (g)-[:has_additional_solvent_percusor]->(s)
    
                """, parameters={"id": gel_id, "additional_solvents_precusors": additional_solvents_precusor}

            )

    modifier = row['Modifier']
    modifier_conc = row['Modifier Concentration (M)']
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
    surfactant_conc = row['Surfactant Concentration (M)']
    if surfactant is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (s:Surfactant {name: $surfactant})
            MERGE (g)-[rel:uses_surfactant]->(s)
                SET rel.surfactant_conc = $surfactant_conc

            """, parameters={"id": gel_id, "surfactant": surfactant, "surfactant_conc": surfactant_conc}

        )

    gelation_agent = row['Gelation Agent']
    gelation_agent_conc = row['Gelation Agent (M)']
    if gelation_agent is not None:
        graph.evaluate(

            """
            MATCH (g:Gel {id: $id})

            MERGE (a:GelationAgent {name: $gelation_agent})
            MERGE (g)-[rel:uses_gelation_agent]->(a)
                SET rel.gelation_agent_conc = $gelation_agent_conc

            """, parameters={"id": gel_id, "gelation_agent": gelation_agent, "gelation_agent_conc": gelation_agent_conc}

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
    wash_duration_1 = row['Wash Duration 1 (days)']
    wash_temp_1 = row['Wash Temp 1 (°C)']
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
    wash_duration_2 = row['Wash Duration 2 (days)']
    wash_temp_2 = row['Wash Temp 2 (°C)']
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
    wash_duration_3 = row['Wash Duration 3 (days)']
    wash_temp_3 = row['Wash Temp 3 (°C)']
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
    wash_duration_4 = row['Wash Duration 4 (days)']
    wash_temp_4 = row['Wash Temp 4 (°C)']
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

    drying_temp_1 = row['Drying Temp (°C)']
    drying_heat_rate_1 = row['Drying Heating Rate (°C/min)']
    drying_pressure_1 = row['Drying Pressure (MPa)']
    drying_time_1 = row['Drying Time (hrs)']
    drying_atmosphere_1 = row['Drying Atmosphere']

    ones = [drying_notes, drying_temp_1, drying_heat_rate_1, drying_pressure_1, drying_time_1, drying_atmosphere_1]

    if any(ones):

        graph.evaluate(
            """
            MATCH (a:Synthesis {id: $id})
            MERGE (d:DryingStep {id: $id, step: 1})
                ON CREATE SET d.notes = $notes, d.drying_temp = $drying_temp, d.drying_heat_rate = $drying_heat_rate,
                        d.drying_pressure = $drying_pressure, d.drying_time = $drying_time, 
                        d.drying_atmosphere = $drying_atmosphere
            MERGE (a)-[:uses_drying_step]->(d)
            """, parameters={"id": gel_id, "notes": drying_notes, "drying_temp": drying_temp_1,
                             "drying_heat_rate": drying_heat_rate_1, "drying_pressure": drying_pressure_1,
                             "drying_time": drying_time_1, "drying_atmosphere": drying_atmosphere_1}
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

    drying_temp_2 = row['Drying Temp 2 (°C)']
    drying_heat_rate_2 = row['Drying Heating Rate 2 (°C/min)']
    drying_pressure_2 = row['Drying Pressure 2 (MPa)']
    drying_time_2 = row['Drying Time 2 (hrs)']
    drying_atmosphere_2 = row['Drying Atmosphere 2']

    twos = [drying_temp_2, drying_heat_rate_2, drying_pressure_2, drying_time_2, drying_atmosphere_2]

    if any(twos):

        graph.evaluate(
            """
            MATCH (a:Synthesis {id: $id})
            MERGE (d:DryingStep {id: $id, step: 2})
                ON CREATE SET d.notes = $notes, d.drying_temp = $drying_temp, d.drying_heat_rate = $drying_heat_rate,
                        d.drying_pressure = $drying_pressure, d.drying_time = $drying_time, 
                        d.drying_atmosphere = $drying_atmosphere
            MERGE (a)-[:uses_drying_step]->(d)
            """, parameters={"id": gel_id, "notes": drying_notes, "drying_temp": drying_temp_2,
                             "drying_heat_rate": drying_heat_rate_2, "drying_pressure": drying_pressure_2,
                             "drying_time": drying_time_2, "drying_atmosphere": drying_atmosphere_2}
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
    df = pd.read_csv('Aerogel.csv')
    df = df.where(pd.notnull(df), None)
    rows = df.to_dict('records')
    for row in rows:
        core_node(row)
        sintering(row)
        lit_info(row)
        gel(row)
        washing_steps(row)
        drying(row)


if __name__ == "__main__":
    # input("Press any button to continue")

    port = 'bolt://localhost:7687'
    username = 'Neo4j'
    password = 'password'
    graph = Graph(port, username=username, password=password)
    main()
