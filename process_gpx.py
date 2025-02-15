import math
import os
import re
from typing import Optional

import polars as pl
import gpxpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def add_leading_zero_to_numbers(s: str) -> str:
    """
    In a string containing text and numbers, reformat every integer
    so it has at least 2 digits (adding a leading zero if needed).

    'Stage 1' -> 'Stage 01'
    'Stage 9' -> 'Stage 09'
    'Stage 10' -> 'Stage 10'
    'Stage 123' -> 'Stage 123'
    """
    return re.sub(
        r"\d+",
        lambda match: f"{int(match.group(0)):02d}",
        s
    )

def gpx_files_to_polars(gpx_dir: str) -> pl.DataFrame:
    """
    Parse all GPX files in a directory and return a Polars DataFrame
    with columns: [filename, longitude, latitude, elevation].
    """
    rows = []

    # List all .gpx files in the directory
    gpx_files = [f for f in os.listdir(gpx_dir) if f.lower().endswith(".gpx")]

    for gpx_file in gpx_files:
        file_path = os.path.join(gpx_dir, gpx_file)

        # Read the GPX file
        with open(file_path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)

        # Iterate through tracks, segments, and points
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    # Append each point to 'rows'
                    rows.append(
                        (
                            add_leading_zero_to_numbers(gpx_file.replace("-route.gpx", "-tdf2024")),
                            point.longitude,  # longitude
                            point.latitude,  # latitude
                            point.elevation,  # elevation (None if not in file)
                        )
                    )

    # Create a Polars DataFrame
    df = pl.DataFrame(
        rows, schema=["course", "longitude", "latitude", "elevation"], orient="row"
    ).sort("course")

    return df


EARTH_RADIUS_M = 6371000  # approximate Earth radius in meters


def haversine_distance(lat1: float, lon1: float, lat2: Optional[float], lon2: Optional[float]) -> Optional[float]:
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
            ), return_dtype=pl.Float64
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
        .with_columns(
            pl.col("elevation").diff().over("course").alias("elevation_diff")
        )
        .with_columns(
            pl.when(pl.col("elevation_diff") > 0)
            .then(pl.col("elevation_diff"))
            .otherwise(pl.lit(0))
            .alias("elevation_gain"),
            (pl.col("elevation_diff") / (pl.col("distance") * 10)).alias("elevation_perc"),
        )
    )

    return df.collect()


def distance_from_start_end(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute the distance from the start and end of the course.
    """
    df = data.with_columns(
        pl.cum_count("distance").over("course").alias("point_index"),
        pl.cum_sum("distance").over("course").alias("distance_from_start"),
    ).sort("course", "point_index", descending=True).with_columns(
        pl.cum_sum("distance").over("course").alias("distance_from_end"),
    ).sort("course", "point_index", descending=False)

    return df

def smooth_data(df: pl.DataFrame, window_size: float) -> pl.DataFrame:
    """Smooth data by creating bins of window_size kilometers"""
    df = df.with_columns((pl.col("distance_from_start") // window_size).alias("distance_binned"))
    df = df.group_by("course", "distance_binned", maintain_order=True).agg(
        pl.mean("elevation", "elevation_perc"),
        pl.last("longitude", "latitude", "distance_from_start"),
        pl.first("distance_from_end"),
        pl.sum("elevation_gain", "distance"),
    ).with_columns(
        pl.col("distance_binned").cast(pl.Int64),
    ).rename({"distance_binned": "point_index"})
    return df


def plot_courses(df: pl.DataFrame):
    df_pd = df.sort("distance_from_end")

    # Make subplots: 2 rows, 1 col, shared X-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,  # share the same x-axis
        vertical_spacing=0.05,  # space between plots
        subplot_titles=("Elevation", "Elevation Gain")  # optional
    )

    colors = px.colors.qualitative.Plotly
    unique_courses = df_pd["course"].unique().to_list()  # polars => list
    course_color_map = {
        c: colors[i % len(colors)]
        for i, c in enumerate(sorted(unique_courses))
    }

    # Add one trace per course in each subplot
    for course, subdf in df_pd.group_by("course", maintain_order=True):
        course_name = course[0]
        subdf = subdf.with_columns(
            pl.col("distance_from_end").round(2),
            pl.col("elevation").round(2),
            pl.col("elevation_gain").round(2),
            pl.col("elevation_perc").round(2),
        )

        # Top row (row=1): Elevation
        fig.add_trace(
            go.Scatter(
                x=subdf["distance_from_end"],
                y=subdf["elevation"],
                mode="lines",
                name=f"{course_name} - elevation",
                line=dict(color=course_color_map[course_name]),
                customdata=subdf["elevation_perc"],  # pass opposite metric
                hovertemplate=(
                        "Course: " + course_name +
                        "<br>Distance to end: %{x}" +
                        "<br>Elevation: %{y}" +
                        "<br>Elevation perc: %{customdata}" +
                        "<extra></extra>"
                ),
            ),
            row=1, col=1
        )

        # Bottom row (row=2): Elevation gain
        fig.add_trace(
            go.Scatter(
                x=subdf["distance_from_end"],
                y=subdf["elevation_perc"],
                mode="lines",
                name=f"{course_name} - elevation_perc",
                line=dict(color=course_color_map[course_name]),
                customdata=subdf["elevation"],  # pass opposite metric
                hovertemplate=(
                        "Course: " + course_name +
                        "<br>Distance to end: %{x}" +
                        "<br>Elevation: %{customdata}" +
                        "<br>Elevation perc: %{y}" +
                        "<extra></extra>"
                ),
            ),
            row=2, col=1
        )

    # Reverse the x-axes (so largest distance is on the left)
    fig.update_xaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(autorange="reversed", row=2, col=1)

    # Set a unified hover so we see one vertical hover line across both plots
    fig.update_layout(
        hovermode="x unified",
        title="Elevation & Elevation Gain vs. Distance from End"
    )

    fig.write_html("elevation_plots.html")


if __name__ == "__main__":
    # Path to the directory where your .gpx files are stored
    gpx_directory = "tdf_2024_gpx"
    data = gpx_files_to_polars(gpx_directory)
    data = distance_between_points(data)
    data = elevation_gains(data)
    data = distance_from_start_end(data)

    smoothed = smooth_data(data, 0.1)

    grouped = data.group_by("course").agg(pl.sum("elevation_gain", "distance"))
    plot_courses(smoothed)
