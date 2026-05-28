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

path = os.path.join("../data_raw")

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
        yearly_flows.to_csv(os.path.join("../data_processed", "yearly_flows" + "_" + year + ".csv"), index=False)

        yearly_flows = yearly_flows[
            yearly_flows["flow_twh"] > 0.1
            ].copy()

        filtered_yearly_flows_list.append(yearly_flows)
        yearly_flows.to_csv(os.path.join("../data_processed", "yearly_flows_filtered" + "_" + year + ".csv"), index=False)

    df = pd.concat(yearly_flows_list, ignore_index=True)
    df.to_csv(os.path.join("../data_processed", "yearly_flows.csv"), index=False)
    filtered_df = pd.concat(filtered_yearly_flows_list, ignore_index=True)
    filtered_df.to_csv(os.path.join("../data_processed", "yearly_flows_filtered.csv"), index=False)