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
    sintering_temp = row['Sintering Temp (°C)']
    sintering_time = row['Sintering Time (min)']
    sintering_ramp_rate = row['Ramp Rate (°C/min)']
    sintering_atmosphere = row['Sintering Atmosphere']
    sintering_notes = row['Sintering Notes']
    graph.evaluate(

        """
        MERGE (n: Aerogel {id: $id})
            ON CREATE SET n.final_product = $final_product, n.pore_volume = $pore_volume, n.pore_size = $pore_size,
                n.nano_particle_size = $nano_particle_size, n.surface_area = $surface_area, n.density = $density,
                n.thermal_conductivity = $thermal_conductivity, n.core_notes = $core_notes,
                n.sintering_temp = $sintering_temp, n.sintering_time = $sintering_time,
                n.sintering_ramp_rate = $sintering_ramp_rate, n.sintering_atmosphere = $sintering_atmosphere,
                n.sintering_notes = $sintering_notes
        
        """, parameters={"id": gel_id, "final_product": final_product, "pore_volume": pore_volume,
                         "pore_size": pore_size, "nano_particle_size": nano_particle_size,
                         "surface_area": surface_area, "density": density, "thermal_conductivity": thermal_conductivity,
                         "core_notes": core_notes, "sintering_temp": sintering_temp, "sintering_time": sintering_time,
                         "sintering_ramp_rate": sintering_ramp_rate, "sintering_atmosphere": sintering_atmosphere,
                         "sintering_notes": sintering_notes}
    )

    porosity = row['Porosity']
    if porosity is not None:
        graph.evaluate(

            """
            MATCH (n: Aerogel {id: $id})
            MERGE (p: Porosity {name: $porosity})
            MERGE (n)-[:has_porosity]->(p)
            """, parameters={"id": gel_id, "porosity": porosity}

        )

    crystalline_phase = row['Crystalline Phase']
    if crystalline_phase is not None:
        graph.evaluate(

            """
            MATCH (n: Aerogel {id: $id})
            MERGE (c: CrystallinePhase {name: $crystalline_phase})
            MERGE (n)-[:has_crystalline_phase]->(c)
            """, parameters={"id": gel_id, "crystalline_phase": crystalline_phase}

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
        MATCH (a:Aerogel {id: $id})
        
        MERGE (lit:LitInfo {id: $title})
            ON CREATE SET lit.author = $author, lit.title = $title, lit.year = $year, 
                lit.cited_reference = $cited_reference, lit.times_cited = $times_cited
        MERGE (a)-[:has_lit_info]->(lit)
        
        """, parameters={"id": gel_id, "author": author, "title": title, "year": year,
                         "cited_reference": cited_reference, "times_cited": times_cited}

    )


def gelation(row):

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
        MATCH (a:Aerogel {id: $id})
        
        MERGE (g:Gelation {id: $id})
            ON CREATE SET g.annas_notes = $annas_notes, g.ph_sol = $ph_sol, g.gelation_temp = $gelation_temp,
                g.gelation_pressure = $gelation_pressure, g.gelation_time = $gelation_time,
                g.aging_time = $aging_time, g.aging_temp = $aging_temp
        MERGE (a)-[:uses_gelation]->(g)
        
        """, parameters={"id": gel_id, "annas_notes": annas_notes,
                         "ph_sol": ph_sol, "gelation_temp": gelation_temp, "gelation_pressure": gelation_pressure,
                         "gelation_time": gelation_time, "aging_time": aging_time, "aging_temp": aging_temp}

    )

    formation_method = row['Formation Method']
    if formation_method is not None:
        graph.evaluate(

            """
            MATCH (g:Gelation {id: $id})
            
            MERGE (f:FormationMethod {name: $formation_method})
            MERGE (g)-[:uses_formation_method]->(f)
            
            """, parameters={"id": gel_id, "formation_method": formation_method}

        )

    zr_precusor = row['Zr Precursor']
    zr_precusor_conc = row['Zr Precursor Concentration (M)']
    if zr_precusor is not None:
        graph.evaluate(

            """
            MATCH (g:Gelation {id: $id})

            MERGE (z:ZrPrecusor {name: $zr_precusor})
                ON CREATE SET z.zr_precusor_conc = $zr_precusor_conc
            MERGE (g)-[:uses_zr_precusor]->(z)

            """, parameters={"id": gel_id, "zr_precusor": zr_precusor, "zr_precusor_conc": zr_precusor_conc}

        )

    dopante = row['Stabilizer/ Dopant/Drying control chemical additive']
    dopante_conc = row['Dopant Concentration (M)']
    if dopante is not None:
        graph.evaluate(

            """
            MATCH (g:Gelation {id: $id})

            MERGE (d:Dopante {name: $dopante})
                ON CREATE SET d.dopante_conc = $dopante_conc
            MERGE (g)-[:uses_dopante]->(d)

            """, parameters={"id": gel_id, "dopante": dopante, "dopante_conc": dopante_conc}

        )

    gel_solvent_1 = row['Solvent 1']
    gel_solvent_1_conc = row['Solvent 1 Concentration (M)']
    if gel_solvent_1 is not None:
        graph.evaluate(

            """
            MATCH (g:Gelation {id: $id})

            MERGE (s:GelSolvent_1 {name: $gel_solvent_1})
                ON CREATE SET s.gel_solvent_1_conc = $gel_solvent_1_conc
            MERGE (g)-[:uses_gel_solvent_1]->(s)

            """, parameters={"id": gel_id, "gel_solvent_1": gel_solvent_1, "gel_solvent_1_conc": gel_solvent_1_conc}

        )

    gel_solvent_2 = row['Solvent 2']
    if gel_solvent_2 is not None:
        graph.evaluate(

            """
            MATCH (g:Gelation {id: $id})

            MERGE (s:GelSolvent_2 {name: $gel_solvent_2})
            MERGE (g)-[:uses_gel_solvent_1]->(s)

            """, parameters={"id": gel_id, "gel_solvent_2": gel_solvent_2}

        )

    additional_solvents_precusors = row['Additional Solvents/Precursors (non Zr)']
    if additional_solvents_precusors is not None:
        graph.evaluate(

            """
            MATCH (g:Gelation {id: $id})

            MERGE (s:AdditionalSolventsAndPrecusors {name: $additional_solvents_precusors})
            MERGE (g)-[:uses_gel_solvent_1]->(s)

            """, parameters={"id": gel_id, "additional_solvents_precusors": additional_solvents_precusors}

        )

    modifier = row['Modifier']
    modifier_conc = row['Modifier Concentration (M)']
    if modifier is not None:
        graph.evaluate(

            """
            MATCH (g:Gelation {id: $id})

            MERGE (m:Modifier {name: $modifier})
                ON CREATE SET m.modifier_conc = $modifier_conc
            MERGE (g)-[:uses_modifier]->(m)

            """, parameters={"id": gel_id, "modifier": modifier, "modifier_conc": modifier_conc}

        )

    surfactant = row['Surfactant']
    surfactant_conc = row['Surfactant Concentration (M)']
    if surfactant is not None:
        graph.evaluate(

            """
            MATCH (g:Gelation {id: $id})

            MERGE (s:Surfactant {name: $surfactant})
                ON CREATE SET s.surfactant_conc = $surfactant_conc
            MERGE (g)-[:uses_surfactant]->(s)

            """, parameters={"id": gel_id, "surfactant": surfactant, "surfactant_conc": surfactant_conc}

        )

    gelation_agent = row['Gelation Agent']
    gelation_agent_conc = row['Gelation Agent (M)']
    if surfactant is not None:
        graph.evaluate(

            """
            MATCH (g:Gelation {id: $id})

            MERGE (a:GelationAgent {name: $gelation_agent})
                ON CREATE SET a.gelation_agent_conc = $gelation_agent_conc
            MERGE (g)-[:uses_surfactant]->(a)

            """, parameters={"id": gel_id, "gelation_agent": gelation_agent, "gelation_agent_conc": gelation_agent_conc}

        )


def washing_steps(row):
    gel_id = row['Index']
    washing_notes = row['Gelation/Washing Notes']

    graph.evaluate(
        """
        MATCH (a:Aerogel {id: $id})
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
            
            MERGE (s:WashingStep {solvent: $wash_solvent})
                ON CREATE SET s.wash_times = $wash_times, s.wash_duration = $wash_duration,
                    s.wash_temp = $wash_temp
            MERGE (w)-[:washing_step_1]->(s)
            
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

            MERGE (s:WashingStep {solvent: $wash_solvent})
                ON CREATE SET s.wash_times = $wash_times, s.wash_duration = $wash_duration,
                    s.wash_temp = $wash_temp
            MERGE (w)-[:washing_step_1]->(s)

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

            MERGE (s:WashingStep {solvent: $wash_solvent})
                ON CREATE SET s.wash_times = $wash_times, s.wash_duration = $wash_duration,
                    s.wash_temp = $wash_temp
            MERGE (w)-[:washing_step_1]->(s)

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

            MERGE (s:WashingStep {solvent: $wash_solvent})
                ON CREATE SET s.wash_times = $wash_times, s.wash_duration = $wash_duration,
                    s.wash_temp = $wash_temp
            MERGE (w)-[:washing_step_1]->(s)

            """, parameters={"id": gel_id, "wash_solvent": wash_solvent_4, "wash_times": wash_times_4,
                             "wash_duration": wash_duration_4, "wash_temp": wash_temp_4}
        )


def drying(row):
    gel_id = row['Index']
    drying_notes = row['Drying Notes']
    drying_temp = row['Drying Temp (°C)']
    drying_heat_rate = row['Drying Heating Rate (°C/min)']
    drying_pressure = row['Drying Pressure (MPa)']
    drying_time = row['Drying Time (hrs)']
    drying_atmosphere = row['Drying Atmosphere']
    graph.evaluate(
        """
        MATCH (a:Aerogel {id: $id})
        
        MERGE (d:Drying {id: $id})
            ON CREATE SET d.notes = $notes, d.drying_temp = $drying_temp, d.drying_heat_rate = $drying_heat_rate,
                d.drying_pressure = $drying_pressure, d.drying_time = $drying_time, 
                d.drying_atmosphere = $drying_atmosphere
        MERGE (a)-[:uses_drying]->(d)
        
        """, parameters={"id": gel_id, "notes": drying_notes, "drying_temp": drying_temp,
                         "drying_heat_rate": drying_heat_rate, "drying_pressure": drying_pressure,
                         "drying_time": drying_time, "drying_atmosphere": drying_atmosphere}
    )

    drying_method = row['Drying Method']
    if drying_method is not None:
        graph.evaluate(
            """
            MATCH (d:Drying {id: $id})
            MERGE (m:DryingMethod {method: $drying_method})
            MERGE (d)-[:uses_drying_method]->(m)
            
            """, parameters={"id": gel_id, "drying_method": drying_method}
        )

    supercritical_solvent = row['Supercritical Solvent']
    if supercritical_solvent is not None:
        graph.evaluate(
            """
            MATCH (d:Drying {id: $id})
            MERGE (scs:SuperCriticalSolvent {solvent: $super_critical_solvent})
            MERGE (d)-[:uses_super_critical_solvent]->(scs)

            """, parameters={"id": gel_id, "super_critical_solvent": supercritical_solvent}
        )

    # drying_temp_2 = row['Drying Temp 2 (°C)']
    # drying_heat_rate_2 = row['Drying Heating Rate 2 (°C/min)']
    # drying_pressure_2 = row['Drying Pressure 2 (MPa)']
    # drying_time_2 = row['Drying Time 2 (hrs)']
    # drying_atmosphere_2 = row['Drying Atmosphere 2']


def main():

    df = pd.read_csv('Aerogel.csv')
    df = df.where(pd.notnull(df), None)
    rows = df.to_dict('records')
    for row in rows:
        core_node(row)
        lit_info(row)
        gelation(row)
        washing_steps(row)
        drying(row)


if __name__ == "__main__":
    port = 'bolt://localhost:7687'
    username = 'Neo4j'
    password = 'password'
    graph = Graph(port, username=username, password=password)
    main()
