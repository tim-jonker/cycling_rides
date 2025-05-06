import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_courses(df: pl.DataFrame):
    df_pd = df.sort("distance_from_end")

    # Make subplots: 2 rows, 1 col, shared X-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,  # share the same x-axis
        vertical_spacing=0.05,  # space between plots
        subplot_titles=("Elevation", "Elevation Gain"),  # optional
    )

    colors = px.colors.qualitative.Plotly
    unique_courses = df_pd["course"].unique().to_list()  # polars => list
    course_color_map = {
        c: colors[i % len(colors)] for i, c in enumerate(sorted(unique_courses))
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
                    "Course: "
                    + course_name
                    + "<br>Distance to end: %{x}"
                    + "<br>Elevation: %{y}"
                    + "<br>Elevation perc: %{customdata}"
                    + "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
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
                    "Course: "
                    + course_name
                    + "<br>Distance to end: %{x}"
                    + "<br>Elevation: %{customdata}"
                    + "<br>Elevation perc: %{y}"
                    + "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    # Reverse the x-axes (so largest distance is on the left)
    fig.update_xaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(autorange="reversed", row=2, col=1)

    # Set a unified hover so we see one vertical hover line across both plots
    fig.update_layout(
        hovermode="x unified", title="Elevation & Elevation Gain vs. Distance from End"
    )

    fig.write_html("elevation_plots.html")


def scatter_plot(df: pl.DataFrame, x_col: str, y_col: str):
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        # color="real_classification",
        title=f"{x_col} vs. {y_col}",
        hover_data=["course"],
    )
    fig.write_html("scatter_plot.html")
