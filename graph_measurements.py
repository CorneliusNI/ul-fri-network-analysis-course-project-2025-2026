import data_extraction 
import networkx as nx
import random
import matplotlib.pyplot as plt


def distances(G, n = 100):
  D = []
  for i in G.nodes() if len(G) <= n else random.sample(list(G), n):
    D.extend([d for d in nx.shortest_path_length(G, source = i).values() if d > 0])
  return D

def info(G):
  print("{:>12s} | '{:s}'".format('Graph', G.name))

  n = G.number_of_nodes()
  m = G.number_of_edges()
  
  print("{:>12s} | {:,d} ({:,d})".format('Nodes', n, nx.number_of_isolates(G)))
  print("{:>12s} | {:,d} ({:,d})".format('Edges', m, nx.number_of_selfloops(G)))
  print("{:>12s} | {:.2f} ({:,d})".format('Degree', 2 * m / n, max([k for _, k in G.degree()])))
  
  if isinstance(G, nx.DiGraph):
    G = nx.MultiGraph(G)

  C = list(nx.connected_components(G))

  print("{:>12s} | {:.1f}% ({:,d})".format('LCC', 100 * max(len(c) for c in C) / n, len(C)))

  D = distances(G)

  print("{:>12s} | {:.2f} ({:,d})".format('Distance', sum(D) / len(D), max(D)))

  if isinstance(G, nx.MultiGraph):
    G = nx.Graph(G)

  print("{:>12s} | {:.4f}".format('Clustering', nx.average_clustering(G)))
  print()
  
  return G

def plot_degrees(G):
  nk = {}
  for _, k in G.degree():
    if k not in nk:
      nk[k] = 0
    nk[k] += 1
  ks = sorted(nk.keys())
  
  plt.loglog(ks, [nk[k] / len(G) for k in ks], '*k')
  plt.title(G.name)
  plt.ylabel('$p_k$')
  plt.xlabel('$k$')
  plt.show()

def draw_graph(G):

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

    plt.title(G.name)
    plt.show()

if __name__ == '__main__':
    G_dict = data_extraction.create_graphs()
    
    for year, G in G_dict.items():
        info(G)
        draw_graph(G)
        plot_degrees(G)

   