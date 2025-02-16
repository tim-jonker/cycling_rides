import math
import os
import re
from tqdm import tqdm
from typing import Optional, List, Dict

import polars as pl
import gpxpy
import overpy
from shapely import LineString
from pyproj import Transformer


def add_leading_zero_to_numbers(s: str) -> str:
    """
    In a string containing text and numbers, reformat every integer
    so it has at least 2 digits (adding a leading zero if needed).

    'Stage 1' -> 'Stage 01'
    'Stage 9' -> 'Stage 09'
    'Stage 10' -> 'Stage 10'
    'Stage 123' -> 'Stage 123'
    """
    return re.sub(r"\d+", lambda match: f"{int(match.group(0)):02d}", s)


def get_surface(longitudes: List[float], latitudes: List[float]) -> Dict[str, float]:
    api = overpy.Overpass()

    # Overpass QL query: find all ways in the bounding box that have a "surface" tag.
    course = LineString([(lon, lat) for lon, lat in zip(longitudes, latitudes)]).buffer(0.01)

    polygon_str = " ".join(f"{lat} {lon}" for lon, lat in course.exterior.coords)
    query = f"""
                    [out:json][timeout:25];
                    (
                      way(poly:"{polygon_str}")["surface"];
                    );
                    out body;
                    >;
                    out skel qt;
                    """
    result = api.query(query)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

    course = LineString([transformer.transform(lon, lat) for lon, lat in zip(longitudes, latitudes)])
    paved_surfaces = {
        "asphalt", "concrete", "paving_stones", "cobblestone",
        "concrete:plates", "sett", "metal", "wood", "unhewn_cobblestone"
    }
    unpaved_surfaces = {
        "gravel", "dirt", "ground", "grass", "sand", "compacted", "fine_gravel"
    }
    surface_category = {surface: "paved" for surface in paved_surfaces}
    surface_category.update({surface: "unpaved" for surface in unpaved_surfaces})
    surface_counts = {}
    for way in result.ways:
        way_line = LineString([transformer.transform(float(n.lon), float(n.lat)) for n in way.nodes])
        intersection_length = way_line.buffer(10, cap_style="flat").intersection(course).length / 1000
        surface = way.tags.get("surface", "unknown")
        surface_cat = surface_category.get(surface, "unknown")
        surface_counts[surface_cat] = surface_counts.get(surface_cat, 0) + intersection_length

    return surface_counts

def gpx_files_to_polars(gpx_dir: str) -> pl.DataFrame:
    """
    Parse all GPX files in a directory and return a Polars DataFrame
    with columns: [filename, longitude, latitude, elevation].
    """
    rows = []

    # List all .gpx files in the directory
    gpx_files = [f for f in os.listdir(gpx_dir) if f.lower().endswith(".gpx")]

    for gpx_file in tqdm(gpx_files):
        file_path = os.path.join(gpx_dir, gpx_file)

        # Read the GPX file
        with open(file_path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)

        # Iterate through tracks, segments, and points
        lats = []
        lons = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    # Append each point to 'rows'
                    rows.append(
                        (
                            add_leading_zero_to_numbers(
                                gpx_file.replace("-route.gpx", "-tdf2024")
                            ),
                            point.longitude,  # longitude
                            point.latitude,  # latitude
                            point.elevation,  # elevation (None if not in file)
                        )
                    )
                    lats.append(point.latitude)
                    lons.append(point.longitude)

        # Get the surface types
        # TODO: Dit is op etappe niveau, niet op segment niveau dus miss is dit niet de beste plek om te verzamelen
        try:
            surface_counts = get_surface(lons, lats)
        except overpy.exception.OverpassRuntimeError as e:
            continue


    # Create a Polars DataFrame
    df = (
        pl.DataFrame(
            rows, schema=["course", "longitude", "latitude", "elevation"], orient="row"
        )
        .sort("course")
        .fill_null(strategy="forward")
    )

    return df


EARTH_RADIUS_M = 6371000  # approximate Earth radius in meters


def haversine_distance(
    lat1: float, lon1: float, lat2: Optional[float], lon2: Optional[float]
) -> Optional[float]:
    """
    Calculate the great-circle distance in kilometers between two points
    specified in decimal degrees (latX, lonX).
    """
    if lat2 is None or lon2 is None:
        return None

    # Convert degrees to radians
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad

    # Haversine formula
    a = (math.sin(d_lat / 2)) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * (
        math.sin(d_lon / 2)
    ) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_M * c

    return distance / 1000


def distance_between_points(df: pl.DataFrame) -> pl.DataFrame:
    """Compute the distance between each point in the DataFrame in kilometers."""
    # Compute the distance between each point
    df = df.with_columns(
        pl.col("latitude").shift(1).over("course").alias("prev_latitude"),
        pl.col("longitude").shift(1).over("course").alias("prev_longitude"),
    )

    df = df.with_columns(
        pl.struct(["latitude", "longitude", "prev_latitude", "prev_longitude"])
        .map_elements(
            lambda x: haversine_distance(
                x["latitude"], x["longitude"], x["prev_latitude"], x["prev_longitude"]
            ),
            return_dtype=pl.Float64,
        )
        .alias("distance")
    ).drop("prev_latitude", "prev_longitude")

    return df


def elevation_gains(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the elevation gain between each point in the DataFrame.
    """
    # Compute the elevation gain between each point
    df = (
        df.lazy()
        .with_columns(pl.col("elevation").diff().over("course").alias("elevation_diff"))
        .with_columns(
            pl.when(pl.col("elevation_diff") > 0)
            .then(pl.col("elevation_diff"))
            .otherwise(pl.lit(0))
            .alias("elevation_gain"),
            (pl.col("elevation_diff") / (pl.col("distance") * 10)).alias(
                "elevation_perc"
            ),
        )
    )

    return df.collect()


def distance_from_start_end(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the distance from the start and end of the course.
    """
    df = (
        df.with_columns(
            pl.cum_count("distance").over("course").alias("point_index"),
            pl.cum_sum("distance").over("course").alias("distance_from_start"),
        )
        .sort("course", "point_index", descending=True)
        .with_columns(
            pl.cum_sum("distance").over("course").alias("distance_from_end"),
        )
        .sort("course", "point_index", descending=False)
    )

    return df


def smooth_data(df: pl.DataFrame, window_size: float) -> pl.DataFrame:
    """Smooth data by creating bins of window_size kilometers"""
    df = df.with_columns(
        (pl.col("distance_from_start") // window_size).alias("distance_binned")
    )
    df = (
        df.group_by("course", "distance_binned", maintain_order=True)
        .agg(
            pl.mean("elevation"),
            pl.last("longitude", "latitude", "distance_from_start"),
            pl.first("distance_from_end"),
            pl.sum("elevation_gain", "distance"),
            (pl.sum("elevation_gain") / (pl.sum("distance") * 10)).alias(
                "elevation_perc"
            ),
        )
        .with_columns(
            pl.col("distance_binned").cast(pl.Int64),
        )
        .rename({"distance_binned": "point_index"})
    )
    return df


def prepare_data(gpx_directory: str) -> pl.DataFrame:
    df = gpx_files_to_polars(gpx_directory)
    df = distance_between_points(df)
    df = elevation_gains(df)
    return distance_from_start_end(df)
