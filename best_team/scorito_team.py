# De link van scorito eindigt op een nummer en dat moet je hieronder invullen op het einde.
# Voor de tour was het 205 en voor vuelta 208, giro 231
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import pulp as plp

from data_reader.read_data import read_scorito_data, read_wielerorakel, read_gpx_file, save_gpx_to_disk, \
    read_all_gpx_of_edition

NUMBER_OF_PAGES = 10
SCORITO_EXTENSION = 283
MULTIPLICATION_LEADER = 2
TEAM_POINTS = 5
BUDGET = 41


def required_skill() -> pd.DataFrame:
    skills = pd.DataFrame(
        [
            [0, 0, 10, -10, 0, 0, 0, 0],  # Sprint
            [10, 0, 2, 0, 0, 0, 0, 0],  # Hills
            [2, 8, 0, 8, 0, 0, 0, 0],  # GCMountain
            [4, 0, 6, -2, 0, 0, 0, 0],  # Sprinthill
            [0, 2, 1, 10, 0, 0, 0, 0],  # GC
            [10, 2, 1, 2, 0, 0, 0, 0],  # HillGC
            [10, 10, 1, -10, 0, 0, 0, 0],  # Breakaway
            [10, 2, 1, -10, 0, 0, 0, 0],  # HillBreakaway
            [2, 10, 1, -10, 0, 0, 0, 0],  # MountainBreakaway
            [0, 0, 0, 0, 0, 10, 0, 0],  # TTShort
            [0, 0, 0, 0, 0, 0, 10, 0],  # TTLong
            [2, 6, 0, 5, 0, 4, 0, 0],  # TTClimb
            [0, 0, 0, 10, 0, 5, 0, 0],  # TTTeam
        ],
        index=["Sprint", "Hills", "GCMountain", "SprintHill", "GC", "HillGC", "Breakaway", "HillBreakaway", "MountainBreakaway",
               "TTShort", "TTLong", "TTClimb", "TTTeam"],
        columns=['hill', 'mountain', 'sprint', 'gc', 'stagespoints', 'ttshort', 'ttlong', 'formtotal']
    )
    skills = skills.div(skills[skills > 0].fillna(0).sum(axis=1), axis=0)
    skills["formtotal"] = 0.1
    return skills


def stage_classification() -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {
            1: "TTShort",
            2: "SprintHill",
            3: "Sprint",
            4: "GC",
            5: "Sprint",
            6: "HillBreakaway",
            7: "SprintHill",
            8: "GC", # Kan een vluchter bij
            9: "GC",
            10: "MountainBreakaway",
            11: "MountainBreakaway",
            12: "GC",
            13: "GC", # Of mountain breakaway
            14: "HillBreakaway", # WvA is hier ook goed
            15: "GC",
            16: "GC",
            17: "Sprint",
            18: "Breakaway",
            19: "GC",
            20: "GC",
            21: "TTShort",
        }, orient="index", columns=["Type"]
    )


def modify_sprint_scores(data: pd.DataFrame, scores: pd.DataFrame, no_mass_sprints: List[str]) -> pd.DataFrame:
    pure_sprint_stages = value.columns[value.loc["Type", :] == "Sprint"]
    all_sprint_stages = value.columns[(value.loc["Type", :] == "Sprint") | (value.loc["Type", :] == "SprintHill")]
    best_sprinter_per_team = list(data.loc[data.groupby('team')['sprint'].idxmax()].index)
    for rider in scores.index:
        # Add points for leadouts
        team = data.at[rider, "team"]
        leadout_points = data.loc[data["team"] == team, "leadout"].sum()
        scores.loc[rider, pure_sprint_stages] += (leadout_points - data.at[rider, "leadout"]) / (7 * 3)
        # Reduce points if not the best sprinter of the team
        if rider not in best_sprinter_per_team:
            scores.loc[rider, all_sprint_stages] /= 2

    scores.loc[no_mass_sprints, all_sprint_stages] = 0
    return scores


def get_stage_scores(value: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    skills = [i for i in value.index if i in data.columns]
    scores = pd.DataFrame(
        np.matmul(data[skills].values, value.loc[skills, :].values),
        index=data.index, columns=value.columns, dtype=float
    )
    return scores


def add_team_points(scores: pd.DataFrame) -> pd.DataFrame:
    leaders = scores.idxmax(axis=0)
    for stage, leader in leaders.items():
        winning_team = data.at[leader, "team"]
        in_winning_team = list(data.index[(data["team"] == winning_team) & (data.index != leader)])
        scores.loc[in_winning_team, stage] += TEAM_POINTS

    return scores


def add_etappe_pred(scores: pd.DataFrame) -> pd.DataFrame:
    et_pred = pd.read_excel('Wielrennen.xlsx', sheet_name="Giro2024Et").fillna(0).set_index("fullName")
    s = scores.merge(et_pred, left_index=True, right_index=True)
    s.iloc[:, :21] += 5 * s.iloc[:, 21:].values
    return s.iloc[:, :21]


def convert_to_scorito_points(column: pd.Series):
    # Sort the column while preserving the original index
    sorted_col = column.sort_values(ascending=False)
    new_values = [1] * len(sorted_col)
    new_values[:20] = [50, 44, 40, 36, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
    value_map = dict(zip(sorted_col, new_values))
    return column.map(value_map)


def calculate_points(scores: pd.DataFrame) -> pd.DataFrame:
    transformed_df = scores.apply(convert_to_scorito_points)
    return transformed_df


def model(scores: pd.DataFrame, data: pd.DataFrame, hardcode_pick: List[str], hardcode_dont_pick: List[str]):
    riders = list(scores.index)
    stages = list(scores.columns)
    model = plp.LpProblem("ScoritoModel", plp.LpMaximize)
    x = {rider: plp.LpVariable(f"Pick {rider}", 0, 1, plp.LpBinary) for rider in riders}
    y = {
        (rider, stage): plp.LpVariable(f"Select {rider} in {stage}", 0, 1, plp.LpContinuous)
        for rider in riders for stage in stages
    }
    z = {
        (rider, stage): plp.LpVariable(f"Leader {rider} in {stage}", 0, 1, plp.LpContinuous)
        for rider in riders for stage in stages
    }

    model += plp.lpSum(
        scores.at[rider, stage] * (y[rider, stage] + z[rider, stage]) for rider in riders for stage in stages)

    for rider in hardcode_pick:
        model.addConstraint(x[rider] == 1, f"Fix {rider}")

    for rider in hardcode_dont_pick:
        model.addConstraint(x[rider] == 0, f"Ignore {rider}")

    for stage in stages:
        for rider in riders:
            model.addConstraint(y[rider, stage] <= x[rider], f"{rider} select in {stage} if picked")
            model.addConstraint(z[rider, stage] <= y[rider, stage], f"{rider} lead in {stage} if selected")

    model.addConstraint(
        plp.lpSum(data.at[rider, "Prijs"] * x[rider] for rider in riders) <= BUDGET, "Budget constraint"
    )

    model.addConstraint(
        plp.lpSum(x[rider] for rider in riders) <= 20, "Max num riders"
    )

    for stage in stages:
        model.addConstraint(plp.lpSum(y[rider, stage] for rider in riders) <= 9, f"Max 9 riders in stage {stage}")
        model.addConstraint(plp.lpSum(z[rider, stage] for rider in riders) <= 1, f"Max 1 leader in stage {stage}")

    for team in data["team"].unique():
        riders_in_team = data.index[data["team"] == team]
        model.addConstraint(
            plp.lpSum(x[rider] for rider in riders_in_team) <= 4, f"Max 4 of {team}"
        )

    model.solve()

    selected_riders = [rider for rider, var in x.items() if plp.value(var) >= 0.5]
    stage_selection = pd.DataFrame().from_dict(
        {
            stage: [rider for rider in riders if plp.value(y[rider, stage]) >= 0.5]
            for stage in stages
        }
    )
    return selected_riders, stage_selection


if __name__ == "__main__":
    # gpx_data = read_all_gpx_of_edition(2023, download=False)

    scorito = read_scorito_data(SCORITO_EXTENSION)
    riders = read_wielerorakel(NUMBER_OF_PAGES)
    combined = scorito.merge(riders, on=["fullName", "dayofbirth"], how="left", validate="1:1")
    combined.to_excel('ScoritoDownload.xlsx', sheet_name='2024')
    missing_riders = combined.loc[combined["average"].isna(), "fullName"]

    data = pd.read_excel('Wielrennen.xlsx', sheet_name='Vuelta2024', index_col=0)
    data.dropna(subset=["average"], inplace=True)
    data.fillna(0, inplace=True)
    data.loc[data["gc"] < 70, "gc"] = 0
    data.loc[data["gc"] >= 70, "sprint"] /= 2
    all_skills = ['flat', 'cobble', 'hill', 'mountain', 'sprint', 'timetrial', 'gc', 'onedaypoints',
                  'stagespoints', 'average', 'leadout', 'ttshort', 'ttlong', 'formtotal']

    skill_minimum = data[data[all_skills] > 0][all_skills].min(axis=0)
    data[all_skills] = (data[all_skills] - skill_minimum) / (100 - skill_minimum) * 100
    data[all_skills] = data[all_skills].mask(data[all_skills] < 0, 0)

    data["gc"] = 20 * data["Klassement"] + 10 * data["gc"]
    data["hill"] = 20 * data["Heuvels"] + 10 * data["hill"]
    data["mountain"] = 20 * data["Klimmen"] + 10 * data["mountain"]
    data["sprint"] = 20 * data["Sprint"] + 10 * data["sprint"]
    data["ttlong"] = 20 * data["Tijdrijden"] + 10 * data["ttlong"]
    data["ttshort"] = 20 * data["Tijdrijden"] + 10 * data["ttshort"]

    # Riders that are not participating
    data = data.loc[
           ~data.index.isin([
               "Elia Viviani",
               "Geraint Thomas",
               "Maxim Van Gils",
               "Caleb Ewan",
               "Juan Ayuso",
               "Stefan Kung",
               "Andrea Vendrame",
               "Max Walscheid",
               "Gerben Thijssen",
               "Luka Mezgec",
               'Kasper Asgreen',
               # 'Mauri Vansevenant',
               # 'Bryan Coquard',
           ]),
           :]

    required_skill = required_skill()
    stage_classification = stage_classification()
    value = stage_classification.merge(required_skill, left_on="Type", right_index=True).sort_index(ascending=True).T
    no_mass_sprints = []
    scores = get_stage_scores(value, data)
    scores = modify_sprint_scores(data, scores, no_mass_sprints)
    scores = add_team_points(scores)
    # scores = add_etappe_pred(scores)
    scores = calculate_points(scores)

    best_performers = {}
    for stage in scores.columns:
        best_performers[stage] = list(scores.sort_values(stage, ascending=False).index)
    best_performers = pd.DataFrame().from_dict(best_performers)

    hardcode_pick = [
        # "Tadej Pogacar", "Jonas Vingegaard", "Jasper Philipsen", "Julian Alaphilippe"
    ]
    hardcode_dont_pick = [
        # "Max Kanter",
        # "Fabio Jakobsen",
        # "Phil Bauhaus",
        # "Filippo Ganna",
        # "Fernando Gaviria",
        # "Jasper Stuyven",
        # "Mikel Landa",
        # "Sepp Kuss",
        # "Mads Pedersen",
        # "Thomas De Gendt",
        # "Vincenzo Albanese",
        # "Robert Gesink",
        # "Giulio Ciccone",
        # "Primoz Roglic",
        # "Einer Rubio",
        # "Clement Berthet",
        # 'Pavel Sivakov',
        # 'Aleksandr Vlasov',
        # 'Attila Valter',
        # 'Kaden Groves',
        # 'Arne Marit',
        # 'Owain Doull',
        # 'Mauri Vansevenant',
        # 'Einer Rubio',
        # 'Dylan Teuns',
    ]
    selected_riders, stage_selection = model(scores, data, hardcode_pick, hardcode_dont_pick)
