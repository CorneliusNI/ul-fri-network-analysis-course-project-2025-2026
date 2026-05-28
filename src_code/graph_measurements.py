import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd

def create_graphs() -> dict:
    df = pd.read_csv("../data_processed/yearly_flows_filtered.csv")

    graphs = {}

    for year in sorted(df["year"].unique()):
        df_year = df[df["year"] == year]
        df_year["distance"] = 1 / df_year["flow_twh"]  # hoher import/export score sorgt für nahe distance
        G = nx.from_pandas_edgelist(
            df_year,
            source="source",
            target="target",
            edge_attr=["flow_twh", "distance"],
            create_using=nx.DiGraph()
        )

        G.name = f"European Electricity Flows {year}"

        graphs[year] = G

    return graphs

def tops(G, C, centrality_name, n=15):
    print("{:>12s} | '{:s}'".format('Centrality', centrality_name))
    sorted_nodes = sorted(
        C.items(),
        key=lambda item: (item[1], G.degree(item[0])),
        reverse=True
    )
    for node, score in sorted_nodes[:n]:
        print(
            "{:>12.6f} | {:>5s} ({:,d})".format(
                score,
                str(node),
                G.degree(node)
            ))
    print()
    
def top_edges(G, C, centrality, n=15):

    print("{:>20s} | {:>12s}".format(
        "Edge",
        centrality
    ))
    # Nach Score sortieren
    sorted_edges = sorted(
        C.items(),
        key=lambda item: item[1],
        reverse=True
    )
    for (source, target), score in sorted_edges[:n]:
        if G.has_edge(source, target):
            flow = G[source][target].get(
                "flow_twh",
                0
            )
        else:
            flow = 0
        print(
            "{:>10s} -> {:<10s} | {:>12.6f} | {:>10.2f} TWh".format(
                source,
                target,
                score,
                flow
            ))
    print()
  
def pagerank_centrality(G, epsilon=1e-6, alpha=0.85):

    # Initialisierung
    P = {node: 1 / len(G) for node in G.nodes()}

    counter = 0

    while True:

        counter += 1

        U = {node: 0 for node in G.nodes()}

        # PageRank Update
        for i in G.nodes():

            for j in G.predecessors(i):

                if G.out_degree(j) > 0:
                    U[i] += alpha * P[j] / G.out_degree(j)

        # Teleportation / Normalisierung
        u = sum(U.values())

        for i in G.nodes():
            U[i] += (1 - u) / len(G)

        # Konvergenztest
        delta = sum(abs(P[i] - U[i]) for i in G.nodes())

        if delta < epsilon:
            break

        P = U

    print("Iterations:", counter)

    return P


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
  
  in_deg = dict(G.in_degree(weight="flow_twh"))
  out_deg = dict(G.out_degree(weight="flow_twh"))

  avg_in = sum(in_deg.values()) / n

  max_in_node = max(in_deg, key=in_deg.get)
  max_out_node = max(out_deg, key=out_deg.get)

  print("{:>12s} | {:.2f} TWh".format(
        "Avg Import/Export",
        avg_in))


  print("{:>12s} | {} ({:.2f} TWh)".format(
        "Top Import:",
      max_in_node,
      in_deg[max_in_node])
    )

  print("{:>12s} | {} ({:.2f} TWh)".format(
        "Top Export",
       max_out_node,
        out_deg[max_out_node]
    ))    

  if isinstance(G, nx.DiGraph):
    G = nx.MultiGraph(G)

  C = list(nx.connected_components(G))

  print("{:>12s} | {:.1f}% ({:,d})".format('LCC', 100 * max(len(c) for c in C) / n, len(C)))

  D = distances(G)

  print("{:>12s} | {:.2f} ({:,d})".format('Distance', sum(D) / len(D), max(D)))

  if isinstance(G, nx.MultiGraph):
    G = nx.Graph(G)

  print("{:>12s} | {:.4f}".format('Clustering', nx.average_clustering(G)))
  
  return G

def top_importers(G, n=15):

    imports = dict(
        G.in_degree(weight="flow_twh")
    )

    ranked = sorted(
        imports.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("Top Importers")

    for node, value in ranked[:n]:
        print(
            f"{node:>5s} "
            f"{value:>10.2f} TWh"
        )

def top_exporters(G, n=10):

    exports = dict(
        G.out_degree(weight="flow_twh")
    )

    ranked = sorted(
        exports.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("Top Exporters")

    for node, value in ranked[:n]:

        print(
            f"{node:>5s} "
            f"{value:>10.2f} TWh"
        )
    
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

def analyze_ukraine_flows(graphs):

    # Alle Ukraine-Varianten
    ua_nodes = {
        "UA",
        "UA-IPS",
        "UA-DobTPP"
    }

    for year, G in graphs.items():

        print("\n" + "=" * 60)
        print(f"UKRAINE ANALYSIS {year}")
        print("=" * 60)

        # Welche Ukraine-Knoten existieren im Graph?
        present_ua = [
            node for node in ua_nodes
            if node in G.nodes()
        ]

        if not present_ua:
            print("No Ukraine nodes found.")
            continue

        total_import = 0
        total_export = 0

        import_countries = {}
        export_countries = {}

        # Alle Ukraine-Knoten einzeln analysieren
        for ua in present_ua:

            print(f"\nNode: {ua}")

            # IMPORTE nach Ukraine
            in_edges = G.in_edges(ua, data=True)

            print("\nImports INTO Ukraine:")

            for source, target, data in in_edges:

                flow = data.get("flow_twh", 0)

                total_import += flow

                if source not in import_countries:
                    import_countries[source] = 0

                import_countries[source] += flow

                print(
                    f"{source:>10s} -> {target:<10s}"
                    f"{flow:>10.2f} TWh"
                )

            # EXPORTE aus Ukraine
            out_edges = G.out_edges(ua, data=True)

            print("\nExports FROM Ukraine:")

            for source, target, data in out_edges:

                flow = data.get("flow_twh", 0)

                total_export += flow

                if target not in export_countries:
                    export_countries[target] = 0

                export_countries[target] += flow

                print(
                    f"{source:>10s} -> {target:<10s}"
                    f"{flow:>10.2f} TWh"
                )

        print("\n" + "-" * 60)

        print(f"Total Import into Ukraine : {total_import:.2f} TWh")
        print(f"Total Export from Ukraine : {total_export:.2f} TWh")

        print("\nMain electricity suppliers to Ukraine:")

        for country, flow in sorted(
            import_countries.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            print(f"{country:>10s}: {flow:>10.2f} TWh")

        print("\nMain electricity receivers from Ukraine:")

        for country, flow in sorted(
            export_countries.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            print(f"{country:>10s}: {flow:>10.2f} TWh")

def compare_betweenness(graphs, weighted=True):

    # Alle Länder sammeln
    all_nodes = set()

    for G in graphs.values():
        all_nodes.update(G.nodes())

    all_nodes = sorted(all_nodes)

    results = {}

    # Für jedes Jahr Betweenness berechnen
    for year, G in graphs.items():

        if weighted:
            bc = nx.betweenness_centrality(
                G,
                weight="distance",
                normalized=True
            )
        else:
            bc = nx.betweenness_centrality(
                G,
                normalized=True
            )

        results[year] = bc

    # Tabelle aufbauen
    table = []

    for node in all_nodes:

        row = {"country": node}

        for year in sorted(graphs.keys()):

            row[year] = round(
                results[year].get(node, 0),
                6
            )

        table.append(row)

    # DataFrame erstellen
    df = pd.DataFrame(table)

    # Nach Ländern sortieren
    df = df.sort_values("country")

    return df

def compare_closeness(graphs, weighted=True):

    # Alle Länder sammeln
    all_nodes = set()

    for G in graphs.values():
        all_nodes.update(G.nodes())

    all_nodes = sorted(all_nodes)

    results = {}

    # Für jedes Jahr Betweenness berechnen
    for year, G in graphs.items():

        if weighted:
            bc = nx.closeness_centrality(
                G,
                distance="distance",
            )
        else:
            bc = nx.closeness_centrality(
                G,
            )

        results[year] = bc

    # Tabelle aufbauen
    table = []

    for node in all_nodes:

        row = {"country": node}

        for year in sorted(graphs.keys()):

            row[year] = round(
                results[year].get(node, 0),
                6
            )

        table.append(row)

    # DataFrame erstellen
    df = pd.DataFrame(table)

    # Nach Ländern sortieren
    df = df.sort_values("country")

    return df


def compare_pagerank(graphs, weighted=True):

    # Alle Länder sammeln
    all_nodes = set()

    for G in graphs.values():
        all_nodes.update(G.nodes())

    all_nodes = sorted(all_nodes)

    results = {}

    # Für jedes Jahr Betweenness berechnen
    for year, G in graphs.items():

        if weighted:
            bc = nx.pagerank(
                G,
                weight="flow_twh",
            )
        else:
            bc = nx.pagerank(
                G,
            )

        results[year] = bc

    # Tabelle aufbauen
    table = []

    for node in all_nodes:

        row = {"country": node}

        for year in sorted(graphs.keys()):

            row[year] = round(
                results[year].get(node, 0),
                6
            )

        table.append(row)

    # DataFrame erstellen
    df = pd.DataFrame(table)

    # Nach Ländern sortieren
    df = df.sort_values("country")

    return df

def compare_import_export(graphs):

    # Alle Länder sammeln
    all_nodes = set()

    for G in graphs.values():
        all_nodes.update(G.nodes())

    all_nodes = sorted(all_nodes)

    rows = []

    # Für jedes Land
    for node in all_nodes:

        row = {
            "country": node
        }

        # Für jedes Jahr
        for year in sorted(graphs.keys()):

            G = graphs[year]

            # Falls Land im Jahr existiert
            if node in G.nodes():

                weighted_import = G.in_degree(
                    node,
                    weight="flow_twh"
                )

                weighted_export = G.out_degree(
                    node,
                    weight="flow_twh"
                )

            else:
                weighted_import = 0
                weighted_export = 0

            row[f"{year}_import"] = round(
                weighted_import,
                3
            )

            row[f"{year}_export"] = round(
                weighted_export,
                3
            )

        rows.append(row)

    # DataFrame erzeugen
    df = pd.DataFrame(rows)

    # Nach Land sortieren
    df = df.sort_values("country")

    return df