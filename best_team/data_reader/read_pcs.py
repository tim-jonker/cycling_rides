import math
from datetime import datetime
from itertools import combinations

from procyclingstats import RiderResults
import polars as pl
import networkx as nx
from pyvis.network import Network

data = pl.read_excel("../Wielrennen.xlsx", sheet_name="Giro2025")
sprinters = data.filter(pl.col("Sprint") > 0).select(["fullName", "Prijs", "Heuvels", "hill", "sprint", "Sprint","formtotal"]).with_columns(
    pl.col("fullName").str.to_lowercase().str.replace_all(" ", "-").alias("rider"),
)
all_riders = sprinters["rider"].to_list()

results = {}
graph = nx.DiGraph()
for rider in all_riders:
    rider_results = RiderResults(f"rider/{rider}/results")
    rider_results.seasons_select()
    x = rider_results.parse()
    results[rider] = (
        pl.from_dicts(x["results"], strict=False)
        .select(["date", "rank", "stage_name", "distance"])
        .drop_nulls()
        .with_columns(
            pl.col("date").str.to_date("%Y-%m-%d").alias("date"),
        )
        .filter((pl.col("date") >= datetime(year=2025, month=1, day=1)) & (pl.col("rank") <= 20))
    )


for rider1, rider2 in combinations(all_riders, 2):
    rider1_results = results[rider1]
    rider2_results = results[rider2]

    merged_results = (
        rider1_results.join(
            rider2_results.select(["date", "rank", "stage_name"]),
            on=["date", "stage_name"],
            suffix="_rider2",
            how="inner",
        )
        .rename({"rank": "rank_rider1"})
        .with_columns(
            pl.when(pl.col("rank_rider1") < pl.col("rank_rider2"))
            .then(1)
            .otherwise(0)
            .alias("rider1_better"),
            pl.when(pl.col("rank_rider1") > pl.col("rank_rider2"))
            .then(1)
            .otherwise(0)
            .alias("rider2_better"),
        )
    )

    if merged_results["rider1_better"].sum() > 0:
        graph.add_edge(rider1, rider2, weight=merged_results["rider1_better"].sum())
    if merged_results["rider2_better"].sum() > 0:
        graph.add_edge(rider2, rider1, weight=merged_results["rider2_better"].sum())


# 1) Compute each node’s total outgoing weight
out_weight = {
    n: sum(data.get("weight", 1) for _, _, data in graph.out_edges(n, data=True))
    for n in graph.nodes()
}

# 2) Sort nodes by descending out_weight
nodes_sorted = sorted(graph.nodes(), key=lambda n: out_weight[n], reverse=True)

N = len(nodes_sorted)
scale = 1000     # makes the circle big enough on screen

# 3) Assign each node a clockwise angle around the circle
#    Highest out_weight → angle = 0 (to the right), then go “down” (clockwise)
pos = {}
for i, n in enumerate(nodes_sorted):
    theta = -2 * math.pi * i / N
    pos[n] = (math.cos(theta) * scale, math.sin(theta) * scale)

net = Network(directed=True, notebook=False, height="800px", width="1000px")
# 2) Let visitors drag nodes around
net.set_options("""
var options = {
  "interaction": {
    "dragNodes": true
  }
}
""")

# 4) Add nodes at their circular coordinates (physics=False so they stay put until dragged)
for node_id, (x, y) in pos.items():
    net.add_node(
        node_id,
        label=str(node_id),
        x=x,
        y=y,
        physics=False,
        font = {"size": 32, "face": "Arial"}
    )
# Put the weight on the edge, both as a title (hover) and as the thickness
seen = set()

for u, v, data in graph.edges(data=True):
    w = data.get("weight", 0)
    pair = (u, v)
    if graph.has_edge(v, u):
        w_rev = graph[v][u].get("weight", 0)
    else:
        w_rev = 0
    rev = (v, u)

    if w > w_rev:
        edge_color = "green"
    elif w < w_rev:
        edge_color = "red"
    else:
        edge_color = "gray"

    # default: straight edge
    smooth_opts = {"enabled": False}

    if rev in graph.edges:
        # if we haven’t yet added the reverse, this is the “first” (curve CW)
        if pair not in seen:
            smooth_opts = {"type": "curvedCW", "roundness": 0.1}
        else:
            # second time we see it, curve CCW
            smooth_opts = {"type": "curvedCCW", "roundness": 0.1}

    net.add_edge(
        u,
        v,
        title=f"Weight: {w}",  # hover label
        value=w,  # thickness ~ weight
        arrows="to",  # draw an arrowhead
        color={"color": edge_color},
        smooth=smooth_opts
    )
    seen.add(pair)

net.write_html("my_graph.html")
