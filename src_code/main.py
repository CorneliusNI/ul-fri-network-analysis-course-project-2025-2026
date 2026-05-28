import os
import networkx as nx
import pandas as pd

from result_path import OUTPUT_PATH
from data_extraction import read_data
from graph_measurements import (
    analyze_ukraine_flows, create_graphs, compare_betweenness, compare_import_export, top_importers, top_exporters,
    info, draw_graph, plot_degrees, tops
)
from main_figure import create_two_panel_network_figure

if __name__ == '__main__':
    ## Process data in data_raw
    read_data()

    ## Perform graph measurements
    G_dict = create_graphs()

    # Analyze betweeness shift
    bc_df = compare_betweenness(G_dict)
    bc_df.to_csv(os.path.join(OUTPUT_PATH, "betweenness.csv"), index=False)
    print(bc_df)

    # Analyze import Export of every country
    in_ex = compare_import_export(G_dict)
    in_ex.to_csv(os.path.join(OUTPUT_PATH, "import_export.csv"), index=False)
    print(in_ex)
    # analyze imports & exports in Ukraine
    analyze_ukraine_flows(G_dict)

    for year, G in G_dict.items():
        info(G)
        top_importers(G, 15)
        top_exporters(G, 15)
    draw_graph(G)
    plot_degrees(G)
    d = nx.degree_centrality(G)
    tops(G, d, f"degree -{year}")
    pr = nx.pagerank(G, weight="flow_twh")
    tops(G, pr, f"pagerank - {year}")
    b = nx.betweenness_centrality(G, weight="distance")
    tops(G, b, f"betweenness - {year}")
    c = nx.closeness_centrality(G, distance="distance")
    tops(G,c, f"closeness - {year}")
    for node in G.nodes(data=True):
        print(node)
    for edge in G.edges(data=True):
        print(edge)

    ## Create main graphic
    df = pd.read_csv("../data_processed/yearly_flows.csv")

    create_two_panel_network_figure(df)