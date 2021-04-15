import pandas as pd
import requests
import json
import time


def fetch_density_molar_mass(compound):

    cached_compound_info = pd.read_csv('cached_compound_info.csv')
    if compound in cached_compound_info['compound'].tolist():
        values = cached_compound_info.loc[cached_compound_info['compound'] == compound].to_dict('records')[0]
        return values['density'], values['molar_mass'], values['cid']

    # Fetch cid number of compund
    cid = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound}/cids/TXT").text
    cid = cid.split()[0]

    # Fetch JSON data of compound
    response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON").text
    data = json.loads(response)

    sections = data['Record']['Section']

    density = ""
    molar_mass = ""
    for section in sections:

        # Look at physical properties
        if section['TOCHeading'] == "Chemical and Physical Properties":
            for subsection in section['Section']:

                # Gather density
                if subsection['TOCHeading'] == "Experimental Properties":
                    for property_type in subsection['Section']:
                        if property_type['TOCHeading'] == 'Density':
                            density_values = property_type['Information']
                            for density_value in density_values:
                                density_test = density_value['Value']['StringWithMarkup'][0]['String']
                                density_test = density_test.split()[0]
                                try:
                                    density = float(density_test)
                                    break
                                except ValueError:
                                    pass
                            break

                # Gather molar mass
                if subsection['TOCHeading'] == "Computed Properties":
                    for property_type in subsection['Section']:
                        if property_type['TOCHeading'] == "Molecular Weight":
                            molar_mass_values = property_type['Information']
                            for molar_mass_value in molar_mass_values:
                                molar_mass = molar_mass_value['Value']['Number'][0]
                                break
                            break

    cached_compound_info = cached_compound_info.append({'compound': compound, 'density': density,
                                                        'molar_mass': molar_mass, 'cid': cid}, ignore_index=True)
    cached_compound_info.to_csv('cached_compound_info.csv', index=False)

    return density, molar_mass, cid


def calculate_molarities(row):

    pore_volume = row['Pore Volume (cm3/g)']
    porosity = row['Porosity']
    if pore_volume == "----" or porosity == "----":
        return None
    pore_volume = float(pore_volume)
    porosity = float(porosity)
    volume = pore_volume/porosity * 0.001  # Convert cc to L

    raw_compounds = row['Material'].split(",")
    raw_compounds = [i.strip() for i in raw_compounds]
    filtered_compounds = []

    molar_ratios = row['Molar Ratio']
    if molar_ratios != "----":

        if row['Molar Ratio 2'] != '----':
            return None

        molar_ratios = molar_ratios.split(",")
        filtered_molar_ratios = []
        for i, molar_ratio in enumerate(molar_ratios):
            molar_ratio = molar_ratio.strip()
            if molar_ratio:
                filtered_compounds.append(raw_compounds[i])
                filtered_molar_ratios.append(float(molar_ratio))
        molar_fractions = [i/sum(filtered_molar_ratios) for i in filtered_molar_ratios]

        molecular_weights = []
        for compound in filtered_compounds:
            density, molar_mass, cid = fetch_density_molar_mass(compound)
            molecular_weights.append(molar_mass)

        total_mass = 0
        for i, molar_fraction in enumerate(molar_fractions):
            molecular_weight = molecular_weights[i]
            total_mass = total_mass + (molar_fraction * molecular_weight)

        mass_fractions = []
        for i, molar_fraction in enumerate(molar_fractions):
            molecular_weight = molecular_weights[i]
            mass_fraction = (molar_fraction * molecular_weight) / total_mass
            mass_fractions.append(mass_fraction)

        molarities = []
        for i, mass_fraction in enumerate(molar_fractions):
            molecular_weight = molecular_weights[i]
            molarity = mass_fraction / (volume * molecular_weight)
            molarities.append(round(molarity, 3))

        compounds = dict(zip(filtered_compounds, molarities))
        return compounds

    elif row['Inittial Volume Added (mL)'] != '----':

        filtered_molarities = []
        if row['Initial Concentration (M)'] != '----':
            starting_molarities = row['Initial Concentration (M)'].split(',')
            filtered_molarities = []
            for starting_molarity in starting_molarities:
                starting_molarity = starting_molarity.strip()
                if starting_molarity:
                    filtered_molarities.append(float(starting_molarity))
                else:
                    filtered_molarities.append("")

        volumes = row['Inittial Volume Added (mL)'].split(',')
        filtered_volumes = []
        for i, volume in enumerate(volumes):
            volume = volume.strip()
            if volume:
                filtered_compounds.append(raw_compounds[i])
                filtered_volumes.append(float(volume))
        volumes = filtered_volumes

        densities = []
        molar_masses = []
        for compound in filtered_compounds:
            density, molar_mass, cid = fetch_density_molar_mass(compound)
            densities.append(density)
            molar_masses.append(molar_mass)

        masses = []
        for i, volume in enumerate(volumes):
            molar_mass = molar_masses[i]
            if filtered_molarities and filtered_molarities[i]:
                mass = (volume * 0.001) * filtered_molarities[i] * molar_mass
                masses.append(mass)
            else:
                density = densities[i]
                mass = volume * density
                masses.append(mass)

        molarities = []
        for i, mass in enumerate(masses):
            molar_mass = molar_masses[i]
            molarity = (mass / molar_mass) / volume
            molarities.append(round(molarity, 3))

        compounds = dict(zip(filtered_compounds, molarities))
        return compounds

    elif row['Initial Mass Added (g)'] != "----":

        masses = row['Initial Mass Added (g)'].split(',')
        filtered_masses = []
        for i, mass in enumerate(masses):
            mass = mass.strip()
            if mass:
                filtered_compounds.append(raw_compounds[i])
                filtered_masses.append(float(mass))
        masses = filtered_masses

        molar_masses = []
        for compound in filtered_compounds:
            density, molar_mass, cid = fetch_density_molar_mass(compound)
            molar_masses.append(molar_mass)

        molarities = []
        for i, mass in enumerate(masses):
            molar_mass = molar_masses[i]
            molarity = (mass / molar_mass) / volume
            molarities.append(round(molarity, 3))

        compounds = dict(zip(filtered_compounds, molarities))
        return compounds

    return None


if __name__ == "__main__":
    df = pd.read_excel('molarity_inputs.xlsx')
    df = df.dropna(how='all', axis=0)

    new_rows = []
    for index, row in df.iterrows():
        row = dict(row)
        new_row = {"Final Material": row['Final Sample']}
        molarities = calculate_molarities(row)
        new_row['calculated_molarities'] = molarities
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    new_df.to_excel('molarity_output.xlsx', index=False)
