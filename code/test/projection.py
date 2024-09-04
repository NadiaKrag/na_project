import networkx as nx
import numpy as np

def assign_attr(U,V,attributes):
    """Assigns attributes from graph U to graph V."""

    for attr in attributes:
        nx.set_node_attributes(V,nx.get_node_attributes(U,attr),name=attr)
    return V

def unweight(G,attributes=list()):
    """Returns the unweighted projected graphs for the sets of vertices in G.

    For unweighted projections, there is an edge between two vertices that
    share a neighbour. All edges are weighted equally (1).

    The square of the adjacency matrix (A_G) of the bipartite graph is calculated
    to give every pair of vertices in one set with at least one shared adjacent
    vertex in the other set a nonzero edge weight. The trace of the resulting
    matrix is removed as self-loops are unwanted. All nonzero elements are then
    set to equal one in accordance to an unweighted projection. A loop through the
    sets of vertices in the bipartite graph is initiated, where the rows and
    columns of A_G corresponding to the vertices in each set is extracted. A graph
    with the proper vertex labels and attributes is then created, appended, and
    returned.
    """

    if not nx.is_bipartite(G):
        raise AssertionError("graph is not bipartite; no projection can be made")

    A_G = nx.adjacency_matrix(G,nodelist=sorted(G.nodes)).todense()
    A_G = A_G ** 2
    A_G = A_G-np.diag(np.diag(A_G))
    A_G[A_G > 0] = 1
    projections = list()
    for U in nx.algorithms.bipartite.basic.sets(G):
        idx = [idx for idx,u in enumerate(sorted(G.nodes)) if u in U]
        A_U = A_G[idx][:,idx]
        labels = {u:v for u,v in enumerate(sorted(U))}
        U = nx.relabel_nodes(nx.from_numpy_matrix(A_U),labels)
        if len(attributes) != 0:
            U = assign_attr(G,U,attributes)
        projections.append(U)
    return tuple(projections)

def simple_weight(G,attributes=list()):
    """Returns the simple-weighted projected graphs for the sets of vertices in G.

    For simple weighted projections, the weight between two vertices correspond
    to the number of neighbours they share in the bipartite graph G.

    The square of the adjacency matrix (A_G) of the bipartite graph is calculated
    to give every pair of vertices in one set an edge weight equal to the number
    of shared adjacent vertices in the other set. The trace of the resulting
    matrix is removed as self-loops are unwanted. A loop through the sets of ver-
    tices in the bipartite graph is initiated, where the rows and columns of A_G
    corresponding to the vertices in each set is extracted. A graph with the pro-
    per vertex labels and attributes is then created, appended, and returned.
    """

    if not nx.is_bipartite(G):
        raise AssertionError("graph is not bipartite; no projection can be made")

    A_G = nx.adjacency_matrix(G,nodelist=sorted(G.nodes)).todense()
    A_G = A_G ** 2
    A_G = A_G-np.diag(np.diag(A_G))
    projections = list()
    for U in nx.algorithms.bipartite.basic.sets(G):
        idx = [idx for idx,i in enumerate(sorted(G.nodes)) if i in U]
        A_U = A_G[idx][:,idx]
        labels = {u:v for u,v in enumerate(sorted(U))}
        U = nx.relabel_nodes(nx.from_numpy_matrix(A_U),labels)
        if len(attributes) != 0:
            U = assign_attr(G,U,attributes)
        projections.append(U)
    return tuple(projections)

def ycn(G,attributes=list()):
    """Returns the YCN-RW projected graphs for the sets of vertices in G.
    """

    if not nx.is_bipartite(G):
        raise AssertionError("graph is not bipartite; no projection can be made")

    projections = list()
    A_G = nx.adjacency_matrix(G,nodelist=sorted(G.nodes)).todense()
    for U in nx.algorithms.bipartite.basic.sets(G):
        idx_U = [idx for idx,i in enumerate(sorted(G.nodes)) if i in U]
        idx_V = [idx for idx,i in enumerate(sorted(G.nodes)) if i not in U]
        T_U = A_G[idx_V][:,idx_U]
        T_U = (T_U.T/np.sum(T_U.T,axis=1)) @ (T_U/np.sum(T_U,axis=1))
        pi = np.ones(T_U.shape[0])/T_U.shape[0]
        pi_temp = np.zeros(T_U.shape[0])
        pi_diff = np.linalg.norm(pi-pi_temp,1)
        i = 0
        while pi_diff > 1e-9 and i < 1e5:
            pi_temp = pi * T_U
            pi = pi_temp * T_U
            pi_diff = np.linalg.norm(pi-pi_temp,1)
            i += 1
        U_U = np.asarray(pi)[0][:,np.newaxis] * np.asarray(T_U)
        U_U = U_U-np.diag(np.diag(U_U))
        labels = {u:v for u,v in enumerate(sorted(U))}
        U_U = nx.relabel_nodes(nx.from_numpy_matrix(U_U),labels)
        if len(attributes) != 0:
            U_U = assign_attr(G,U_U,attributes)
        projections.append(U_U)
    return tuple(projections)


# OLD ALGORITHMS BELOW

def unweight_old(G,attributes=list()):
    """Returns the unweighted projected graphs for the sets of vertices in G.
    For unweighted projections, there is an edge between two vertices that
    share a neighbour. All edges are weighted equally (1).
    """

    if not nx.is_bipartite(G):
        raise AssertionError("graph is not bipartite; no projection can be made")

    U,V = nx.algorithms.bipartite.basic.sets(G)
    w_U,w_V = nx.Graph(),nx.Graph()
    for i in U:
        for j in U:
            if i == j:
                continue
            w_U.add_edge(i,j)
    for i in V:
        for j in V:
            if i == j:
                continue
            w_V.add_edge(i,j)
    if len(attributes) !=0:
        w_U,w_V = assign_attr(G,w_U,attributes),assign_attr(G,w_V,attributes)
    return w_U,w_V

def simple_weight_old(G,attributes=list()):
    """Returns the simple weighted projected graphs for the sets of vertices in G.

    For simple weighted projections, the weight between two vertices correspond
    to the number of neighbours they share in the bipartite graph G.
    """

    if not nx.is_bipartite(G):
        raise AssertionError("graph is not bipartite; no projection can be made")

    U,V = nx.algorithms.bipartite.basic.sets(G)
    w_U,w_V = nx.Graph(),nx.Graph()
    for i in U:
        for j in U:
            if i == j:
                continue
            weight = len(set(G[i]) & set(G[j]))
            if weight == 0:
                continue
            w_U.add_edge(i,j,weight=weight)
    for i in V:
        for j in V:
            if i == j:
                continue
            weight = len(set(G[i]) & set(G[j]))
            if weight == 0:
                continue
            w_V.add_edge(i,j,weight=weight)
    if len(attributes) !=0:
        w_U,w_V = assign_attr(G,w_U,attributes),assign_attr(G,w_V,attributes)
    return w_U,w_V
