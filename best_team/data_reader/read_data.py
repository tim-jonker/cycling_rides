import json
from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd
import requests
from tqdm import tqdm
from unidecode import unidecode
import gpxpy
import gpxpy.gpx


def read_scorito_data(scorito_extension: int) -> pd.DataFrame:
    link = f'https://cycling.scorito.com/cyclingmanager/v1.0/eventriderenriched/{scorito_extension}'
    response = requests.get(
        link, headers={
        'User-Agent': 'curl/8.6.0'
        }
    )
    riders = json.loads(response.text)['Content']
    for idx, rider in enumerate(riders):
        for key in rider['Qualities']:
            riders[idx]["Type " + str(key['Type'])] = key['Value']
    all_riders = pd.DataFrame(riders)
    all_riders.rename(columns={"BirthDate": "dayofbirth"}, inplace=True)
    all_riders["dayofbirth"] = all_riders["dayofbirth"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S").date())
    all_riders["fullName"] = all_riders["FirstName"] + " " + all_riders["LastName"]
    all_riders["fullName"].replace({"-": " ", "_": " "}, regex=True, inplace=True)
    all_riders["fullName"] = all_riders["fullName"].apply(lambda x: unidecode(x))
    all_riders.rename(columns={'Type 0': 'Klassement',
                               'Type 1': 'Klimmen',
                               'Type 2': 'Tijdrijden',
                               'Type 3': 'Sprint',
                               'Type 4': 'Punch',
                               'Type 5': 'Heuvels',
                               'Price': 'Prijs',
                               'TeamId': 'Team'}, inplace=True)
    all_riders['Prijs'] /= 1000000
    all_riders = all_riders[["fullName", "dayofbirth", 'Prijs', 'Klassement', 'Klimmen', 'Tijdrijden', 'Sprint', 'Punch', 'Heuvels']]
    all_riders.fillna(0, inplace=True)
    return all_riders


def read_wielerorakel(pages: int) -> pd.DataFrame:
    all_riders = pd.DataFrame()
    for page in tqdm(range(1, pages + 1)):
        query_string = r'''query FetchRiders($take: Int!, $page: Int!, $gender: Gender!, $search: String!, $orderBy: OrderByDto) {
                fetchRiders(take: $take, page: $page, gender: $gender, search: $search, orderBy: $orderBy) {
                    count
                    riders {
                        id
                        fullName
                        slug
                        firstName
                        nation
                        length
                        weight
                        dayofbirth
                        currentTeam {
                            id
                            name
                            slug
                        }
                        currentStats {
                            flat
                            cobble
                            hill
                            mountain
                            sprint
                            timetrial
                            gc
                            onedaypoints
                            stagespoints
                            average
                            leadout

                            ttshort
                            ttlong
                            formtotal
                        }
                    }
                }
            }'''

        payload = {
            "query": query_string,
            "variables": {
                "take": 100,
                "page": page,
                "gender": "MEN",
                "search": "",
                "orderBy": {
                    "field": "fullName",
                    "orderBy": "ASC",
                    "nestedField": None
                }
            }
        }
        response = requests.post(
            url="https://api.cyclingoracle.com/v1",
            headers={
                "accept": "application/json, text/javascript, */*; q=0.01",
                "accept-language": "en-GB,en-US;q=0.9,en;q=0.8,nl;q=0.7",
                "content-type": "application/json",
                "sec-ch-ua": "\"Google Chrome\";v=\"113\", \"Chromium\";v=\"113\", \"Not-A.Brand\";v=\"24\"",
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": "\"macOS\"",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-site",
                "Referer": "https://www.cyclingoracle.com/nl/renners",
                "Referrer-Policy": "no-referrer-when-downgrade",
                "x-api-key": "c81823a3-ea7e-4a48-97ab-aa3372fd1a0b",
            },
            json=payload,
        )
        a = json.loads(response.content)
        riders = a["data"]["fetchRiders"]["riders"]
        if not riders:
            continue
        for rider in riders:
            for quality, value in rider["currentStats"].items():
                rider[quality] = value
            rider["team"] = rider["currentTeam"]["name"]
            del rider["currentStats"]
            del rider["currentTeam"]
        df = pd.DataFrame(riders)
        df = df[['fullName', 'nation', 'length', 'weight',
                 'dayofbirth', 'flat', 'cobble', 'hill', 'mountain', 'sprint',
                 'timetrial', 'gc', 'onedaypoints', 'stagespoints', 'average', 'leadout',
                 'ttshort', 'ttlong', 'formtotal', 'team']]
        all_riders = pd.concat([all_riders, df])

    all_riders["dayofbirth"] = all_riders["dayofbirth"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z").date())
    all_riders["fullName"] = all_riders["fullName"].apply(lambda x: unidecode(x))
    all_riders["fullName"].replace({"-": " ", "_": " "}, regex=True, inplace=True)
    all_riders["fullName"].replace(
        {
            "Tom Pidcock": "Thomas Pidcock",
            "Mikkel Honore": "Mikkel Frolich Honore",
            "Edward Dunbar": "Eddie Dunbar",
            "Sebastian Molano": "Juan Sebastian Molano",
            "Daniel Martinez": "Daniel Felipe Martinez",
            "Lucas Plapp": "Luke Plapp",
            "Cristian Scaroni": "Christian Scaroni",
            "Isaac Del Toro": "Isaac del Toro",
            "Mattias Skjelmose": "Mattias Skjelmose Jensen",
        }, inplace=True
    )
    return all_riders


@dataclass
class Point:
    latitude: float
    longitude: float
    elevation: float


def read_gpx_file(file_name: str) -> List[Point]:
    gpx_file = open(file_name, 'r')
    gpx = gpxpy.parse(gpx_file)
    all_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                all_points.append(Point(point.latitude, point.longitude, point.elevation))
    return all_points


def save_gpx_to_disk(year: int, stage: int, download: bool):
    url = f'https://cdn.touretappe.nl/images/tour-de-france/{year}/etappe-{stage}-route.gpx'
    filename = f'gpx_files/etappe-{year}-{stage}.gpx'

    if download:
        response = requests.get(url)

        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            return filename
        else:
            raise ConnectionError("Failed to download the GPX file.")
    else:
        return filename


def read_all_gpx_of_edition(year: int, download: bool =False) -> List[List[Point]]:
    all_stages = []
    for stage in tqdm(range(1, 22)):
        filename = save_gpx_to_disk(year, stage, download=download)
        all_stages.append(read_gpx_file(filename))
    return all_stages
