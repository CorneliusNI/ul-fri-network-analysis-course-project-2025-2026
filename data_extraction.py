import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import os

country_codes = [
    "AL",
    "AM",
    "AZ",
    "BA",
    "BG",
    "CZ",
    "GE",
    "GR",
    "HR",
    "HU",
    "MD",
    "ME",
    "MK",
    "PL",
    "RO",
    "RS",
    "RU_KGD",
    "SL",
    "SK",
    "UA_IPS",
    "XK"
]

years_of_interest = ["2019", "2020", "2021", "2022", "2023", "2024"]

path = os.path.join("data_raw")

def read_data() -> None:
    yearly_flows_list = []
    filtered_yearly_flows_list = []

    for year in years_of_interest:
        list_of_dfs = []
        for country in country_codes:
            file_name = "GUI_NET_CROSS_BORDER_PHYSICAL_FLOWS_" + year + "_" + country + ".csv"
            df = pd.read_csv(os.path.join(path, year, file_name))
            list_of_dfs.append(df)

        df = pd.concat(list_of_dfs, ignore_index=True)

        df = df.drop_duplicates()

        df = df.rename(columns={
            "MTU": "time",
            "Out Area": "source",
            "In Area": "target",
            "Physical Flow (MW)": "flow_mw"
        })
        # Extract first timestamp
        df["start_time"] = df["time"].str.split(" - ").str[0]

        # Remove optional timezone suffixes like (CET) or (CEST)
        df["start_time"] = df["start_time"].str.replace(
            r" \(CET\)| \(CEST\)",
            "",
            regex=True
        )

        # Convert to datetime
        df["start_time"] = pd.to_datetime(
            df["start_time"],
            format="%d/%m/%Y %H:%M:%S"
        )

        df["year"] = df["start_time"].dt.year

        df["source"] = df["source"].str.replace("BZN|", "", regex=False)
        df["target"] = df["target"].str.replace("BZN|", "", regex=False)

        df = df.dropna(subset=["source", "target", "flow_mw"])
        df = df[df["source"] != df["target"]]

        df["flow_mw"] = pd.to_numeric(df["flow_mw"], errors="coerce")
        df = df.dropna(subset=["flow_mw"])

        yearly_flows = (
            df.groupby(["year", "source", "target"])["flow_mw"]
            .sum()
            .reset_index()
        )
        yearly_flows = yearly_flows.sort_values(
            by="flow_mw",
            ascending=False
        )
        yearly_flows["flow_twh"] = yearly_flows["flow_mw"] / 1_000_000

        yearly_flows_list.append(yearly_flows)
        yearly_flows.to_csv(os.path.join("data_processed", "yearly_flows" + "_" + year + ".csv"), index=False)

        yearly_flows = yearly_flows[
            yearly_flows["flow_twh"] > 0.1
            ].copy()

        filtered_yearly_flows_list.append(yearly_flows)
        yearly_flows.to_csv(os.path.join("data_processed", "yearly_flows_filtered" + "_" + year + ".csv"), index=False)

    df = pd.concat(yearly_flows_list, ignore_index=True)
    df.to_csv(os.path.join("data_processed", "yearly_flows.csv"), index=False)
    filtered_df = pd.concat(filtered_yearly_flows_list, ignore_index=True)
    filtered_df.to_csv(os.path.join("data_processed", "yearly_flows_filtered.csv"), index=False)

def create_graph() -> nx.DiGraph:
    df = pd.read_csv("data_processed/yearly_flows_filtered.csv")
    df_2024 = df[df["year"] == 2024]
    #df_2024 = df_2024[df_2024["flow_twh"] >= 2]
    G = nx.from_pandas_edgelist(
        df_2024,
        source="source",
        target="target",
        edge_attr="flow_twh",
        create_using=nx.DiGraph()
    )
    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(G, seed=42)

    weights = [
        G[u][v]["flow_twh"]
        for u, v in G.edges()
    ]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1200,
        font_size=9,
        width=[w / 2 for w in weights],
        arrows=True
    )

    plt.title("European Electricity Flows 2024")
    plt.show()
    return G

def create_map_graph():
    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    # Your processed yearly flows

    df = pd.read_csv("data_processed/yearly_flows_filtered.csv")

    # Select one year
    YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

    # Filter year
    for year in YEARS:

        df_copy = df[df["year"] == year]

        # Keep only important flows
        # Adjust threshold as needed

        df_copy = df_copy[df_copy["flow_twh"] >= 2]

        # --------------------------------------------------
        # OPTIONAL CLEANUP
        # --------------------------------------------------

        # Merge Ukrainian variants
        mapping = {
            "UA-IPS": "UA",
            "UA-DobTPP": "UA"
        }

        # Apply mapping

        df_copy["source"] = df_copy["source"].replace(mapping)
        df_copy["target"] = df_copy["target"].replace(mapping)

        # Re-aggregate after merging

        df_copy = (
            df_copy.groupby(["source", "target"], as_index=False)["flow_twh"]
            .sum()
        )

        # --------------------------------------------------
        # COUNTRY COORDINATES
        # --------------------------------------------------

        # Approximate geographic coordinates
        # You can refine these later.

        positions = {
            "AT": (14.5, 47.5),
            "BA": (17.8, 44.2),
            "BG": (25.3, 42.7),
            "CZ": (15.5, 49.8),
            "DE-LU": (10.5, 51.0),
            "GR": (22.0, 39.0),
            "HR": (16.5, 45.2),
            "HU": (19.0, 47.0),
            "IT-North": (10.5, 45.0),
            "IT-South": (16.0, 41.0),
            "IT-Centre-South": (13.5, 42.5),
            "LT": (24.0, 55.2),
            "MD": (28.5, 47.0),
            "ME": (19.3, 42.8),
            "MK": (21.7, 41.6),
            "PL": (19.0, 52.0),
            "RO": (25.0, 45.8),
            "RS": (20.8, 44.0),
            "SE4": (14.0, 56.0),
            "SI": (14.8, 46.1),
            "SK": (19.5, 48.7),
            "TR": (35.0, 39.0),
            "UA": (31.0, 49.0),
            "XK*": (20.9, 42.7),
            "AL": (20.0, 41.2),
            "AZ": (47.5, 40.5),
            "GE": (43.5, 42.2),
            "AM": (44.5, 40.3),
            "RU": (37.0, 55.0),
            "RU-KGD": (20.5, 54.7)
        }

        # --------------------------------------------------
        # BUILD GRAPH
        # --------------------------------------------------

        G = nx.DiGraph()

        for _, row in df_copy.iterrows():
            source = row["source"]
            target = row["target"]
            flow = row["flow_twh"]

            if source in positions and target in positions:
                G.add_edge(source, target, weight=flow)

        # --------------------------------------------------
        # LOAD EUROPE MAP
        # --------------------------------------------------

        world = gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        )

        # Keep Europe + nearby regions

        europe = world[
            world["CONTINENT"].isin(["Europe", "Asia"])
        ]

        # --------------------------------------------------
        # PLOT
        # --------------------------------------------------

        fig, ax = plt.subplots(figsize=(18, 14))

        # Draw map background

        europe.plot(
            ax=ax,
            color="whitesmoke",
            edgecolor="gray"
        )

        # Draw nodes

        nx.draw_networkx_nodes(
            G,
            positions,
            node_size=600,
            node_color="red",
            ax=ax
        )

        # Draw labels

        nx.draw_networkx_labels(
            G,
            positions,
            font_size=9,
            font_weight="bold",
            ax=ax
        )

        # Edge widths scaled by flow size

        weights = [
            G[u][v]["weight"]
            for u, v in G.edges()
        ]

        scaled_widths = [w / 2 for w in weights]

        # Draw directed edges

        nx.draw_networkx_edges(
            G,
            positions,
            width=scaled_widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20,
            edge_color="royalblue",
            alpha=0.7,
            connectionstyle="arc3,rad=0.05",
            ax=ax
        )

        # --------------------------------------------------
        # FINAL FORMATTING
        # --------------------------------------------------

        ax.set_title(
            f"European Electricity Flows ({year})",
            fontsize=20,
            fontweight="bold"
        )

        ax.set_xlim(-10, 50)
        ax.set_ylim(35, 65)

        ax.set_axis_off()

        fig_name = "electricity_flows_" + str(year) + ".png"
        plt.tight_layout()
        plt.savefig(
            os.path.join("results", fig_name),
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

if __name__ == '__main__':
    #read_data()
    #create_graph()
    create_map_graph()