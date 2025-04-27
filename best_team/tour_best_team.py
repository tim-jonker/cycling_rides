import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import pulp as plp

KLASSEMENTEN = ['Alg.', 'Punt', 'Berg', 'Jong']
ETAPPES = 21
year = 2023
BUDGET = 51

team_points = {'Etappe': 10, 'Alg.': 8, 'Punt': 6, 'Berg': 6, 'Jong': 3}

etappe_uitslag = pd.read_excel(f'C:\\Users\\timjo\\OneDrive\\Tim\\TourDeFrance\\Ideale Tourselectie {year}.xlsx', sheet_name='Top20')
points_per_etappe = pd.read_excel(f'C:\\Users\\timjo\\OneDrive\\Tim\\TourDeFrance\\Ideale Tourselectie {year}.xlsx', sheet_name='Points')
riders = pd.read_excel(f'C:\\Users\\timjo\\OneDrive\\Tim\\TourDeFrance\\Ideale Tourselectie {year}.xlsx', sheet_name='Riders')
klassement = pd.read_excel(f'C:\\Users\\timjo\\OneDrive\\Tim\\TourDeFrance\\Ideale Tourselectie {year}.xlsx', sheet_name='Klassement')

etappe_uitslag.dropna(inplace=True)
klassement.dropna(inplace=True, subset=['Alg.'])
klassement.fillna('Unaffordable', inplace=True)
riders = riders._append({'Team': 'UNA', 'Name': 'Unaffordable', 'Cost (mln)': BUDGET}, ignore_index=True)
etappe_uitslag['Etappe'] = etappe_uitslag['Etappe'].astype(int)
points_per_etappe.rename(columns={'Etappe': 'Points'}, inplace=True)
klassement.rename(columns={'positie': 'Positie', 'etappe': 'Etappe'}, inplace=True)

etappe_uitslag = etappe_uitslag.merge(points_per_etappe[['Positie', 'Points']], on='Positie')
klassement = klassement.merge(points_per_etappe[['Positie'] + KLASSEMENTEN], on='Positie')

points = pd.DataFrame({'Renner': list(riders['Name']) * ETAPPES, 'Etappe': np.repeat(range(1, ETAPPES+1), len(riders))})
points = points.merge(etappe_uitslag, how='outer', on=['Renner', 'Etappe'])
for klas in KLASSEMENTEN:
    points = points.merge(klassement[['Etappe', klas + '_x', klas + '_y']], how='outer', left_on=['Etappe', 'Renner'], right_on=['Etappe', klas + '_x'])
    points.drop(columns=[klas + '_x'], inplace=True)

points['Points'] = points[['Points'] + [klas + '_y' for klas in KLASSEMENTEN]].sum(axis=1)
points['Points'] = points['Points'].astype(int)
points = points.filter(['Etappe', 'Renner', 'Points'])
points.dropna(inplace=True)

points_pivot = points.pivot_table(values='Points', index='Renner', columns='Etappe')
points_pivot = points_pivot.merge(riders, left_index=True, right_on='Name')
points_pivot.set_index('Name', inplace=True)
points_pivot = points_pivot[['Cost (mln)', 'Team'] + list(range(1, ETAPPES+1))]

for etappe in range(1, ETAPPES + 1):
    winnaar = etappe_uitslag.loc[(etappe_uitslag['Etappe'] == etappe) & (etappe_uitslag['Positie'] == 1), 'Renner'].iat[0]
    winnende_ploeg = points_pivot.at[winnaar, 'Team']
    points_pivot.loc[points_pivot['Team'] == winnende_ploeg, etappe] += team_points['Etappe']
    points_pivot.at[winnaar, etappe] -= team_points['Etappe']

    for klas in KLASSEMENTEN:
        winnaar = klassement.loc[(klassement['Etappe'] == etappe) & (klassement['Positie'] == 1), klas+'_x'].iat[0]
        winnende_ploeg = points_pivot.at[winnaar, 'Team']
        points_pivot.loc[points_pivot['Team'] == winnende_ploeg, etappe] += team_points[klas]
        points_pivot.at[winnaar, etappe] -= team_points[klas]

model = plp.LpProblem("Maximize points", plp.LpMaximize)

select_rider = plp.LpVariable.dicts("select", (rider for rider in points_pivot.index), lowBound=0, upBound=1, cat='Integer')
assignment = plp.LpVariable.dicts("select", ((rider, etappe) for rider in points_pivot.index for etappe in range(1, ETAPPES+1)), lowBound=0, upBound=1, cat='Continuous')

model += plp.lpSum(points_pivot.at[renner, etappe] * assignment[(renner, etappe)] for renner in points_pivot.index for etappe in range(1, ETAPPES+1))

# Total amount of renners
model += plp.lpSum(val for val in select_rider.values()) == 20

# Total amount of budget
model += plp.lpSum(points_pivot.at[renner, 'Cost (mln)'] * val for renner, val in select_rider.items()) <= BUDGET

# Can only select chosen renners
for renner in points_pivot.index:
    for etappe in range(1, ETAPPES+1):
        model += assignment[(renner, etappe)] <= select_rider[renner]

# Maximally 9 renners per etappe
for etappe in range(1, ETAPPES+1):
    model += plp.lpSum(assignment[(renner, etappe)] for renner in points_pivot.index) == 9

# Max 4 of one team
team_dict = riders[['Name', 'Team']]
team_dict.set_index('Name', inplace=True)
for team in riders['Team'].unique():
    for etappe in range(1, ETAPPES + 1):
        model += plp.lpSum(assignment[(renner, etappe)] for renner in points_pivot.index if team_dict.at[renner, 'Team'] == team) <= 4

model.solve(use_mps=False)

select = [rider for rider, value in select_rider.items() if value.varValue > 0.1]

per_etappe = pd.DataFrame()
for etappe in range(1, ETAPPES+1):
    pos = 0
    for renner in points_pivot.index:
        if assignment[(renner, etappe)].varValue > 0.5:
            per_etappe.at[pos, etappe] = renner
            pos += 1

used_budget = riders.loc[riders['Name'].isin(select), 'Cost (mln)'].sum()
points = model.objective.value()
