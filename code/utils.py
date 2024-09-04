import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import infomap
from collections import Counter
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import k_clique_communities


class CommunityDiscovery:
    def infomap_detect(self,G,parameters=list(),com_name="infomap"):
        """
        Partition network with the Infomap algorithm.
        Take a networkx graph as input and return communities.
        """
        
        # Create an Infomap instance with given parameters
        parameters.append("--silent")
        space = " "
        infomapWrapper = infomap.Infomap(space.join(parameter for parameter in parameters))
        
        # Access the default network to add links
        network = infomapWrapper.network()
        
        # Add links and weight as optional third argument
        edges = nx.get_edge_attributes(G,"weight")
        for e in edges:
            x,y = e
            network.addLink(int(x),int(y),int(edges[e]))
        
        # Run the Infomap search algorithm to find optimal modules
        infomapWrapper.run()

        # Tree node iterator
        communities = {}
        for node in infomapWrapper.iterTree(1): # https://mapequation.github.io/infomap/#itertree
            if node.isLeaf():
                communities[node.physicalId] = node.moduleIndex()

        print("Found {} modules with codelength: {}".format(infomapWrapper.numTopModules(), infomapWrapper.codelength()))

        nx.set_node_attributes(G,name=com_name,values=communities)

        return G,communities

    def greedy(self,G,com_name="greedy"):
        coms = greedy_modularity_communities(G)
        communities = dict()
        for idx,com in enumerate(coms):
            for node in com:
                communities[node] = idx

        nx.set_node_attributes(G,name=com_name,values=communities)

        return G,communities

    def k_clique (self,G,k,com_name="k_clique"):
        coms = k_clique_communities(G,k)
        communities = dict()
        seen_coms = []
        last_idx = 0
        for idx,com in enumerate(coms):
            for node in com:
                communities[node] = idx
                seen_coms.append(node)
            last_idx = idx
        no_com = [node for node in list(G.nodes) if node not in seen_coms]
        last_idx += 1
        for node in no_com:
            communities[node] = last_idx
        nx.set_node_attributes(G,name=com_name,values=communities)
        return G, communities

class Helper:
    def drawCommunities(self,G,ax,com_name= "community",proj_name="projection"):
        pos = nx.spring_layout(G,weight="weight",iterations=25)

        communities = [community for community in nx.get_node_attributes(G,com_name).values()]
        numCommunities = len(set(communities))

        c = plt.get_cmap("tab20")
        lights = mpl.colors.ListedColormap([c(i)[:3] for i in range(c.N) if i % 2],"indexed",numCommunities)
        darks = mpl.colors.ListedColormap([c(i)[:3] for i in range(c.N) if not i % 2],"indexed",numCommunities)
        
        # draw edges and colors of vertices
        nx.draw_networkx_edges(G,pos)
        nodeCollection = nx.draw_networkx_nodes(G,
                                                ax=ax,
                                                pos=pos,
                                                node_color=communities,
                                                cmap=lights)

        # set edge colors
        darkColors = [darks(v) for v in communities]
        nodeCollection.set_edgecolor(darkColors)

        # give vertices labels and colors
        for idx,n in enumerate(G.nodes()):
            plt.annotate(n,
                        xy=pos[n],
                        textcoords="offset points",
                        horizontalalignment="center",
                        verticalalignment="center",
                        xytext=[0,0],
                        color=darks(communities[idx] % 10))
        plt.title("{}_{} has {} numbers of communities".format(proj_name,com_name,numCommunities))
        plt.axis("off")

    def stack_plot(self,G,com):
        """
        Creates a stack plot two show the distributions of subclasses in each discorvered
        community (com)
        Parameters: 
        G: The graph
        com: the name of the community
        """
        hatchs = ['/', 'O','|', '-', '+', 'x', 'o', '.', '*'] *3
        NUM_COLORS = 22
        cm = plt.get_cmap('gist_rainbow')
        plot_dict = {}
        subclass = []
        for node in G.nodes():
            plot_dict.setdefault(G.nodes[node][com],[]) 
            plot_dict[G.nodes[node][com]].append(G.nodes[node]['Subclass'])
            subclass.append(G.nodes[node]['Subclass'])
        subclass = list(set(subclass))
        mat = np.zeros((len(set(subclass)),len(plot_dict.keys())))
        
        #print(plot_dict)
        for col in range(mat.shape[1]):
            for row in range(mat.shape[0]):
                #print(col,row)
                counter = plot_dict[list(plot_dict.keys())[col]].count(subclass[row])
                mat[row,col] = counter
        
        N = mat.shape[1]
        ind = np.arange(N)    # the x locations for the groups
        width = 0.85       # the width of the bars: can also be len(x) sequence
        bars = []
        for dis in range(mat.shape[0]):
            bars.append(plt.barh(ind, mat[dis,:], width,color = cm(dis//1*1.0/NUM_COLORS),hatch =hatchs[dis]))
            

        plt.ylabel('Community')
        plt.title('Distribution from %s'%com)
        plt.yticks(ind, [col for col in range(mat.shape[1])])
        #ticks = np.arange(0, 35, 1) if com != "Subclass" else np.arange(0, 100, 5)
        #plt.xticks(ticks)
        plt.legend([bar[0] for bar in bars], subclass)
        plt.show()



class Similarity:
    def norm_mutual_info(self,C1,C2):
        """
        Takes two divisions (dictionary with vertices as keys and communities
        as values) of a network and returns the normalized mutual information.
        Each division is assumed to be zero-based-numbered.
        """

        # Amount of nodes in each partition
        assert len(C1.keys()) == len(C2.keys())
        N = len(C1.keys())

        # Reverse partition dictionaries
        C1_inv = dict()
        for vertex,community in C1.items():
            C1_inv[community] = C1_inv.get(community,list())
            C1_inv[community].append(vertex)
        
        C2_inv = dict()
        for vertex,community in C2.items():
            C2_inv[community] = C2_inv.get(community,list())
            C2_inv[community].append(vertex)


        mutual_information = 0
        for c1 in set(C1.values()):
            for c2 in set(C2.values()):
                joint = len(set(C1_inv[c1]) & set(C2_inv[c2])) / N
                if joint == 0:
                    continue
                mutual_information += joint * np.log2(joint / (len(C1_inv[c1]) * len(C2_inv[c2])/N**2))
        
        shannon_entropy_C1 = -sum([len(c)/N * np.log2(len(c)/N) for c in C1_inv.values()])
        shannon_entropy_C2 = -sum([len(c)/N * np.log2(len(c)/N) for c in C2_inv.values()])
        return 2 * mutual_information / (shannon_entropy_C1 + shannon_entropy_C2)
        


class Projection:
    def assign_attr(self,U,V,attributes):
        """
        Takes two networkx graphs (U and V) and a list of strings (attributes) as input,
        assigns attributes from U to V, and returns V.
        """

        for attr in attributes:
            nx.set_node_attributes(V,nx.get_node_attributes(U,attr),name=attr)
        return V

    def unweight(self,G,attributes=list()):
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
                U = self.assign_attr(G,U,attributes)
            projections.append(U)
        return tuple(projections)

    def simple_weight(self,G,attributes=list()):
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
                U = self.assign_attr(G,U,attributes)
            projections.append(U)
        return tuple(projections)
    
    def ycn(self,G,attributes=list()):
        """Returns the YCN-RW projected graphs for the sets of vertices in G."""

        if not nx.is_bipartite(G):
            raise AssertionError("graph is not bipartite; no projection can be made")

        projections = list()
        A_G = nx.adjacency_matrix(G,nodelist=sorted(G.nodes))
        for U in nx.algorithms.bipartite.basic.sets(G):
            idx_U = [idx for idx,i in enumerate(sorted(G.nodes)) if i in U]
            idx_V = [idx for idx,i in enumerate(sorted(G.nodes)) if i not in U]
            T_U = A_G[idx_U][:,idx_V]
            T_U = (T_U/np.sum(T_U,axis=1)) @ (T_U.T/np.sum(T_U.T,axis=1))
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
            labels = {idx:v for idx,v in enumerate(sorted(U))}
            G_U = nx.relabel_nodes(nx.from_numpy_matrix(U_U),labels)

            # normalize weights after lowest
            G_U_w = nx.get_edge_attributes(G_U,name="weight")
            G_U_w_min = min(G_U_w.values())
            G_U_w_scaled = dict()
            for edge,weight in G_U_w.items():
                G_U_w_scaled[edge] = weight/G_U_w_min
            nx.set_edge_attributes(G_U,G_U_w_scaled,name="weight")

            if len(attributes) != 0:
                G_U = self.assign_attr(G,G_U,attributes)
            projections.append(G_U)
        return tuple(projections)



if __name__ == "__main__":
    pass
