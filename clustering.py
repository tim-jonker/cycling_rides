from typing import List, Optional

import polars as pl

from process_gpx import prepare_data, smooth_data
from visualization import scatter_plot
from sklearn.cluster import KMeans

import overpy
from shapely import LineString
from pyproj import Transformer

def get_surface(longitudes: List[float], latitudes: List[float]) -> float:
    """Returns the fraction of unpaved road in the course."""
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
    try:
        result = api.query(query)
    except overpy.exception.OverpassRuntimeError:
        return 0

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

    course = LineString([transformer.transform(lon, lat) for lon, lat in zip(longitudes, latitudes)])
    paved_surfaces = {
        "asphalt", "concrete", "paving_stones",
        "concrete:plates", "sett", "metal", "wood",
    }
    unpaved_surfaces = {
        "gravel", "dirt", "ground", "grass", "sand", "compacted", "fine_gravel"
    }
    cobble_surfaces = {
        "unhewn_cobblestone", "cobblestone"
    }
    surface_category = {surface: "paved" for surface in paved_surfaces}
    surface_category.update({surface: "unpaved" for surface in unpaved_surfaces})
    surface_category.update({surface: "cobble" for surface in cobble_surfaces})
    surface_counts = {}
    for way in result.ways:
        way_line = LineString([transformer.transform(float(n.lon), float(n.lat)) for n in way.nodes])
        intersection_length = way_line.buffer(10, cap_style="flat").intersection(course).length / 1000
        surface = way.tags.get("surface", "unknown")
        surface_cat = surface_category.get(surface, "unknown")
        surface_counts[surface_cat] = surface_counts.get(surface_cat, 0) + intersection_length

    return surface_counts.get("unpaved", 0) / (surface_counts.get("paved", 1) + surface_counts.get("unpaved", 0))


def standard_scale(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    """
    Standard scales specified columns: (value - mean) / std.
    Adds new columns with suffix '_standard_scaled'.
    """
    for col in cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        # Create a new column for the scaled data
        df = df.with_columns(
            ((pl.col(col) - mean_val) / std_val).alias(f"{col}_standard_scaled")
        )
    return df

def robust_scale(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    """
    Robust scales specified columns: (value - median) / IQR.
    Adds new columns with suffix '_robust_scaled'.
    """
    for col in cols:
        median_val = df[col].median()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df = df.with_columns(
            ((pl.col(col) - median_val) / iqr).alias(f"{col}_robust_scaled")
        )
    return df


def group_data(data: pl.DataFrame, window_size: Optional[float]) -> pl.DataFrame:
    if window_size:
        data = smooth_data(data, window_size)
    # plot_courses(smoothed)
    unclassified = data.group_by("course", maintain_order=True).agg(
        pl.sum("elevation_gain", "distance"), pl.col("longitude"), pl.col("latitude")
    )
    return (
        unclassified
        .with_columns(
            pl.struct('longitude', 'latitude')
            .map_elements(lambda r: get_surface(r['longitude'], r['latitude']), return_dtype=pl.Float64)
            .alias('f_unpaved')
        )
    )


if __name__ == "__main__":
    load_data = True
    gpx_directory = "tdf_2024_gpx"
    data = prepare_data(gpx_directory)

    if load_data:
        unclassified = pl.read_parquet("unclassified.parquet")
    else:
        unclassified = group_data(data, None)
        unclassified.write_parquet("unclassified.parquet")

    unclassified = unclassified.with_columns(
        real_classification=pl.Series(
            [
                "hill",
                "hill",
                "flat",
                "mountain",
                "flat",
                "flat",
                "tt",
                "hill",
                "gravel",
                "flat",
                "hill",
                "hill",
                "hill",
                "mountain",
                "mountain",
                "flat",
                "hill",
                "hill",
                "mountain",
                "mountain",
                "tt",
            ],
            strict=False,
        )
    )
    for last_x in [30, 10, 1]:
        last_xk = data.filter(pl.col("distance_from_end") <= last_x)
        last_xk = last_xk.group_by("course", maintain_order=True).agg(
            pl.sum("elevation_diff").alias(f"elevation_diff_last{last_x}"),
            pl.sum("elevation_gain").alias(f"elevation_gain_last{last_x}")
        )
        unclassified = unclassified.join(last_xk, on="course", how="left")

    for first_x in [50]:
        first_xk = data.filter(pl.col("distance_from_start") <= first_x)
        first_xk = first_xk.group_by("course", maintain_order=True).agg(
            pl.sum("elevation_diff").alias(f"elevation_diff_first{first_x}"),
            pl.sum("elevation_gain").alias(f"elevation_gain_first{first_x}")
        )
        unclassified = unclassified.join(first_xk, on="course", how="left")

    unclassified = unclassified.with_columns(
        (pl.col("elevation_gain") / pl.col("distance")).alias("gradient"),
    )
    num_clusters = 7

    unclassified = unclassified.with_columns(pl.when(pl.col("distance") <= 60).then(pl.lit(num_clusters - 1)).when(pl.col("f_unpaved") > 0.1).then(pl.lit(num_clusters - 2)).otherwise(pl.lit(0)).alias("predicted_labels"))
    classified = unclassified.filter(pl.col("predicted_labels") > 0)
    unclassified = unclassified.filter(pl.col("predicted_labels") == 0).drop("predicted_labels")


    scatter_plot(unclassified, "distance", "gradient")

    x = unclassified.select(pl.col(pl.NUMERIC_DTYPES))
    x = standard_scale(x, x.columns)
    kmeans = KMeans(n_clusters=num_clusters - classified["predicted_labels"].n_unique(), random_state=0, n_init="auto").fit_predict(x)
    unclassified = unclassified.with_columns(predicted_labels=pl.Series(kmeans))

    classified = pl.concat([classified.select("course", "real_classification", "predicted_labels"), unclassified.select("course", "real_classification", "predicted_labels")]).sort("course")

    # TODO:
    #  Count number of climbs, average climb length
