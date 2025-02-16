from typing import List

import polars as pl

from process_gpx import prepare_data
from visualization import scatter_plot
from sklearn.cluster import KMeans


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

if __name__ == "__main__":
    # Path to the directory where your .gpx files are stored
    gpx_directory = "tdf_2024_gpx"
    data = prepare_data(gpx_directory)

    # smoothed = smooth_data(data, 0.5)
    # plot_courses(smoothed)

    unclassified = data.group_by("course", maintain_order=True).agg(
        pl.sum("elevation_gain", "distance")
    )
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

    unclassified = unclassified.with_columns(
        (pl.col("elevation_gain") / pl.col("distance")).alias("gradient"),
    )
    num_clusters = 6

    classified = unclassified.filter(pl.col("distance") <= 60).with_columns(predicted_labels=pl.lit(num_clusters - 1))
    unclassified = unclassified.filter(pl.col("distance") > 60)


    scatter_plot(unclassified, "distance", "gradient")

    x = unclassified.select(pl.col(pl.NUMERIC_DTYPES))
    x = standard_scale(x, x.columns)
    kmeans = KMeans(n_clusters=num_clusters - 1, random_state=0, n_init="auto").fit_predict(x)
    unclassified = unclassified.with_columns(predicted_labels=pl.Series(kmeans))

    classified = pl.concat([classified.select("course", "real_classification", "predicted_labels"), unclassified.select("course", "real_classification", "predicted_labels")]).sort("course")

    # TODO:
    #  Count number of climbs, average climb length
    #  Climb in first 50km is more likely for breakaways
