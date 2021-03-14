import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import scipy


def combined_graphs_edges(bigGraph, smallGraph):
    for u, v, data in smallGraph.edges(data=True):
        w = data['weight']
        if bigGraph.has_edge(u, v):
            bigGraph[u][v][0]['weight'] += w
        else:
            bigGraph.add_edge(u, v, weight=w)

    return bigGraph


def degree_histogram_directed(G, in_degree=False, out_degree=False):
    """Return a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq


def plot_degree_graph(freq, title):
    plt.figure(figsize=(12, 8))
    plt.loglog(range(len(freq)), freq, 'bo-')
    plt.xlabel(title)
    plt.ylabel('Frequency')

    num = '(a) '
    if title == 'Indegree':
        num = '(b) '
    else:
        if title == 'Outdegree':
            num = '(c) '

    plt.title(num + title)
    plt.savefig('./' + title + '.png')


def plot_degree_dist(G):
    in_degree_freq = degree_histogram_directed(G, in_degree=True)
    out_degree_freq = degree_histogram_directed(G, out_degree=True)
    degree_freq = degree_histogram_directed(G)

    plot_degree_graph(in_degree_freq, 'Indegree')
    plot_degree_graph(out_degree_freq, 'Outdegree')
    plot_degree_graph(degree_freq, 'Degree')


def page_rank(G):
    pk = nx.pagerank_numpy(G, weight=None)
    # return heapq.nlargest(10, pk.items(), key=lambda x: x[1])
    return max(pk.items(), key=lambda x: x[1])


def degree_centrality(G):
    dc = nx.degree_centrality(G)
    # return heapq.nlargest(10, dc.items(), key=lambda x: x[1])
    return max(dc.items(), key=lambda x: x[1])


if __name__ == '__main__':
    usecols = ['from_address', 'to_address', 'value']
    dirname = input('Please enter full path to the transactions folder\n')
    os.chdir(dirname)

    g = nx.MultiDiGraph()

    for start_block in os.listdir(dirname):
        end_block = os.listdir(start_block)[0]
        filename = os.listdir(start_block + '/' + end_block)[0]
        pip = start_block + '/' + '/' + end_block + '/' + filename
        print(pip)

        p = pd.read_csv(
            pip,
            usecols=usecols,
            error_bad_lines=False,
            index_col=False,
            dtype='unicode',
            low_memory=False,
        )
        p = p[p.value != 0]
        p = p.dropna()
        p['weight'] = p.groupby(['from_address', 'to_address'])['value'].transform('sum')
        p.drop_duplicates(subset=['from_address', 'to_address'], inplace=True)
        gc = nx.convert_matrix.from_pandas_edgelist(
            df=p,
            source='from_address',
            target='to_address',
            edge_attr='weight',
            create_using=nx.MultiDiGraph(),
        )
        g = combined_graphs_edges(g, gc)

    dc = nx.degree_centrality(g)
    nx.set_node_attributes(g, dc, 'degreeCentrality')
    print('Most important node by degree centrality:', max(dc.items(), key=lambda x: x[1]))

    g = nx.convert_node_labels_to_integers(g)
    print('Number of accounts:', g.number_of_nodes())
    plot_degree_dist(g)

    diG = nx.Graph(g)
    print('Clustering coefficient:', nx.average_clustering(diG))

    print('Pearson coefficient:', nx.degree_pearson_correlation_coefficient(g))
    print('SCC/WCC:')
    print('     Largest SCC size:', len(max(nx.strongly_connected_components(g), key=len)))
    print('     Number of SCCs:', nx.number_strongly_connected_components(g))
    print('     Number of WCCs:', nx.number_weakly_connected_components(g))

    #    print('Top 10 most important nodes in MFG, ranked by PageRank')
    #    print(page_rank(g), '\n')
    #    print('Top 10 most important nodes in MFG, ranked by degree centrality')
    #    print(degree_centrality(g), '\n')

    print('Assorativity coefficient:', nx.attribute_assortativity_coefficient(g, 'degreeCentrality'))
