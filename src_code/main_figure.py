import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from result_path import OUTPUT_PATH


# =========================================================
# CONFIG
# =========================================================

MIN_FLOW = 5
TOP_N = 35
FIGSIZE = (20, 10)

MAP_XLIM = (-12, 48)
MAP_YLIM = (35, 62)

EDGE_ALPHA = 0.55
EDGE_RAD = 0.12

NODE_SIZE = 40
NODE_FONT = 8

BASEMAP_COLOR = "#f5f5f5"
BASEMAP_EDGE = "gray"

# =========================================================
# NODE COORDINATES
# =========================================================

COORDS = {
    "DE-LU": (10.5, 51.2),
    "PL": (19.1, 52.1),
    "CZ": (15.5, 49.8),
    "SK": (19.5, 48.7),
    "HU": (19.0, 47.1),
    "AT": (14.5, 47.5),
    "SI": (14.9, 46.1),
    "HR": (16.4, 45.65),
    "RO": (24.9, 45.9),
    "BG": (25.2, 42.7),
    "GR": (22.0, 39.3),
    "RS": (20.8, 44.0),
    "BA": (17.8, 44.2),
    "ME": (19.3, 42.8),
    "MK": (21.7, 41.6),
    "AL": (20.0, 41.2),
    "IT-North": (10.5, 45.0),
    "IT-South": (16.5, 40.6),
    #"IT-CSouth": (14.3, 42.0),
    #"UA": (25.0, 49.0),
    "UA-IPS": (35.0, 49.2),
    "LT": (24.0, 55.2),
    "SE4": (13.5, 56.0),
    "TR": (35.0, 39.0),
    "GE": (43.5, 42.0),
    "AZ": (47.5, 40.3),
    "MD": (28.5, 47.2),
    "XK*": (20.9, 42.7),
    "AM": (44.8, 40.2),
    "RU": (37.6, 55.7),
    "RU-KGD": (20.5, 54.7),
    "BY": (27.9, 53.7)
}


# =========================================================
# DATA PIPELINE
# =========================================================

def aggregate_period(df, start_year, end_year):
    period = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    return (
        period.groupby(["source", "target"])["flow_twh"]
        .sum()
        .reset_index()
    )


def make_undirected(df):
    """
    Merge bidirectional flows into a single undirected edge.
    Ensures (A,B) == (B,A).
    """

    df = df.copy()

    # canonical ordering
    df["node_a"] = df[["source", "target"]].min(axis=1)
    df["node_b"] = df[["source", "target"]].max(axis=1)

    # aggregate BOTH directions safely
    undirected = (
        df.groupby(["node_a", "node_b"], as_index=False)
          .agg({"flow_twh": "sum"})
    )

    return undirected.rename(columns={
        "node_a": "source",
        "node_b": "target"
    })


def filter_network(df, min_flow=MIN_FLOW, top_n=TOP_N):
    df = df[df["flow_twh"] >= min_flow]
    return df.nlargest(top_n, "flow_twh")


# =========================================================
# MAP SETUP (CARTOPY)
# =========================================================

def setup_axis(ax):
    ax.add_feature(cfeature.LAND, facecolor=BASEMAP_COLOR)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=BASEMAP_EDGE)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    ax.set_extent([MAP_XLIM[0], MAP_XLIM[1], MAP_YLIM[0], MAP_YLIM[1]])


# =========================================================
# EDGE DRAWING
# =========================================================

FLOW_COLOR_BINS = [0, 2, 5, 10, np.inf]
FLOW_COLORS = [
    "#56B4E9",  # light sky blue
    "#009E73",  # green
    "#0072B2",  # blue
    "#003F5C"  # deep blue-black
]

def get_flow_color(flow):
    """
    Assign discrete color based on flow magnitude.
    """
    for i in range(len(FLOW_COLOR_BINS) - 1):
        if FLOW_COLOR_BINS[i] <= flow < FLOW_COLOR_BINS[i + 1]:
            return FLOW_COLORS[i]
    return FLOW_COLORS[-1]

def draw_edges(ax, df, max_flow):
    """
    Draw curved network edges with discrete color bins.
    """

    for _, row in df.iterrows():

        src = row["source"]
        tgt = row["target"]

        if src not in COORDS or tgt not in COORDS:
            continue

        lon1, lat1 = COORDS[src]
        lon2, lat2 = COORDS[tgt]

        flow = row["flow_twh"]

        # thickness (continuous)
        linewidth = 1 + 4 * np.sqrt(flow / max_flow)

        # color (discrete bins)
        color = get_flow_color(flow)

        edge = FancyArrowPatch(
            (lon1, lat1),
            (lon2, lat2),
            connectionstyle=f"arc3,rad={EDGE_RAD}",
            linewidth=linewidth,
            alpha=EDGE_ALPHA,
            arrowstyle="-",
            color=color,
            zorder=2
        )

        ax.add_patch(edge)


# =========================================================
# NODE DRAWING
# =========================================================

def draw_nodes(ax):

    for node, (lon, lat) in COORDS.items():

        ax.scatter(
            lon,
            lat,
            transform=ccrs.PlateCarree(),
            s=NODE_SIZE,
            color="black",
            edgecolors="white",
            linewidth=0.7,
            zorder=5
        )

        ax.text(
            lon + 0.4,
            lat + 0.2,
            node,
            transform=ccrs.PlateCarree(),
            fontsize=NODE_FONT,
            zorder=6
        )


# =========================================================
# PANEL
# =========================================================

def draw_network_panel(ax, network_df, max_flow, title):

    setup_axis(ax)

    draw_edges(ax, network_df, max_flow)
    draw_nodes(ax)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_axis_off()


# =========================================================
# LEGEND
# =========================================================

def add_shared_legend(ax):
    """
    Legend for discrete flow color bins.
    """

    handles = [
        Line2D([0], [0], color=FLOW_COLORS[0], lw=4, label="0–5 TWh"),
        Line2D([0], [0], color=FLOW_COLORS[1], lw=4, label="5–10 TWh"),
        Line2D([0], [0], color=FLOW_COLORS[2], lw=4, label="10–15 TWh"),
        Line2D([0], [0], color=FLOW_COLORS[3], lw=4, label="15+ TWh"),
    ]

    legend = ax.legend(
        handles=handles,
        title="Annual flow",
        loc="upper left",
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )

    ax.add_artist(legend)

def add_flow_width_legend(ax, max_flow):
    """
    Legend for edge width scaling.
    """

    example_flows = [5, 10, 20]

    handles = []

    for f in example_flows:
        lw = 0.5 + 6 * np.sqrt(f / max_flow)

        handles.append(
            Line2D(
                [0], [0],
                color="black",
                lw=lw,
                label=f"{f} TWh (width)"
            )
        )

    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=8,
        title="Line width",
        title_fontsize=9
    )


# =========================================================
# MAIN FIGURE
# =========================================================

def create_two_panel_network_figure(
    df,
    output_path=os.path.join(OUTPUT_PATH, "main_fig.png")
):

    # -------------------------
    # aggregate periods
    # -------------------------
    pre = aggregate_period(df, 2019, 2021)
    post = aggregate_period(df, 2022, 2024)

    # -------------------------
    # undirected network
    # -------------------------
    pre = make_undirected(pre)
    post = make_undirected(post)

    # -------------------------
    # filtering
    # -------------------------
    #pre = filter_network(pre)
    #post = filter_network(post)

    # -------------------------
    # shared scaling
    # -------------------------
    max_flow = max(pre["flow_twh"].max(), post["flow_twh"].max())

    # -------------------------
    # figure
    # -------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=FIGSIZE,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    draw_network_panel(
        axes[0],
        pre,
        max_flow,
        "Aggregated Cross-Border Electricity Flows (2019–2021)"
    )

    draw_network_panel(
        axes[1],
        post,
        max_flow,
        "Aggregated Cross-Border Electricity Flows (2022–2024)"
    )

    add_shared_legend(fig)
    #add_flow_width_legend(fig, max_flow)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {output_path}")