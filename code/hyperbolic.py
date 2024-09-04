def hyperbolic(G,attributes=list()):
    if not nx.is_bipartite(G):
        raise AssertionError("graph is not bipartite; no projection can be made")

    M = nx.MultiGraph()
    hdn = nx.Graph()

    A_G = nx.adjacency_matrix(G,nodelist=sorted(G.nodes)).todense()
    for U in nx.algorithms.bipartite.basic.sets(G):
        idx_U = [idx for idx,i in enumerate(sorted(G.nodes)) if i in U]
        idx_V = [idx for idx,i in enumerate(sorted(G.nodes)) if i not in U]
        labels = {idx:v for idx,v in enumerate(sorted(G.nodes)) if v not in U}
        T_U = A_G[idx_U][:,idx_V]
    dis = []      
    for row in range(len(T_U)):
        dis.append([j for idx, j in labels.items() if T_U[row,idx]==1])
    T_U = [1/row.sum() for i, row in enumerate(T_U)]
    i_TU = 0
    for row in dis:
        ind = 0
        for i in row:
            ind = ind + 1
            for j in row[ind:]:
                if i != j:
                    w = T_U[i_TU]
                    M.add_edge(i,j, weight = w)
        i_TU = i_TU + 1
    for i,j,data in M.edges(data=True):
        w = data['weight'] if 'weight' in data else 0.0
        if hdn.has_edge(i,j):
            hdn[i][j]['weight'] = hdn[i][j]['weight'] + w
        else:
            hdn.add_edge(i, j, weight = w)

    if len(attributes) != 0:
        for attr in attributes:
            nx.set_node_attributes(hdn,nx.get_node_attributes(G,attr),name=attr)
    
    return hdn