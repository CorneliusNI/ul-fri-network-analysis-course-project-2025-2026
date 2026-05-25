import data_extraction 
import networkx as nx

import networkx as nx

def info(G):
    print("{:>12s} | '{:s}'".format('Graph', G.name))

    n = G.number_of_nodes()
    m = G.number_of_edges()

    print("{:>12s} | {:,d} ({:,d})".format('Nodes', n, nx.number_of_isolates(G)))
    print("{:>12s} | {:,d} ({:,d})".format('Edges', m, nx.number_of_selfloops(G)))


    if G.is_directed():
        deg = sum(dict(G.degree()).values()) / n
        max_deg = max(dict(G.degree()).values())
    else:
        deg = 2 * m / n
        max_deg = max(dict(G.degree()).values())

    print("{:>12s} | {:.2f} ({:,d})".format('Degree', deg, max_deg))

    
    if G.is_directed():
        C = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    else:
        C = sorted(nx.connected_components(G), key=len, reverse=True)

    print("{:>12s} | {:.1f}% ({:,d})".format(
        'LCC', 100 * len(C[0]) / n, len(C))
    )

    # clustering only works with undirected graphs
    if G.is_directed():
        Gu = G.to_undirected()
    else:
        Gu = G

    print("{:>12s} | {:.4f}".format('Clustering', nx.average_clustering(Gu)))
    print()


if __name__ == '__main__':
    G = data_extraction.create_graph()
    G.name = "European Electricity Flows 2024"
    info(G)