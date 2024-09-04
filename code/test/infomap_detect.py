import infomap
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

def infomap_detect(G):
    """
    Partition network with the Infomap algorithm.
    Take a networkx graph as input, annotates nodes with 'community' id
    and return number of communities found.
    """
    
    #Create an Infomap instance
    infomapWrapper = infomap.Infomap("--two-level \
                                      --silent \
                                      --num-trials 10 \
                                      --teleportation-probability 0.15 \
                                      --markov-time 1 \
                                      --core-loop-limit 10 \
                                      --tune-iteration-threshold 1e-5")
    
    #Access the default network to add links
    network = infomapWrapper.network()
    
    #Add links and weight as optional third argument
    edges = nx.get_edge_attributes(G,"weight")
    for e in edges:
        x,y = e
        network.addLink(int(x),int(y),int(edges[e]))
    
    # Run the Infomap search algorithm to find optimal modules
    print("Find communities with Infomap...")
    infomapWrapper.run()

    print("Found {} modules with codelength: {}".format(infomapWrapper.numTopModules(), infomapWrapper.codelength()))

    #Tree node iterator
    communities = {}
    for node in infomapWrapper.iterTree():
        if node.isLeaf():
            communities[node.physicalId] = node.moduleIndex()

    print("Actually found {} communities.".format(len(set(communities.values()))))

    nx.set_node_attributes(G, name='community', values=communities)
    return infomapWrapper.numTopModules(), communities, G

def drawNetwork(G):
    # create figure
    fig = plt.figure(figsize=(21,13))
    ax = fig.add_subplot(111)
    # position map
    pos = nx.spring_layout(G,weight="weight",iterations=250)
    # community ids
    communities = [community for vertex,community in nx.get_node_attributes(G,"community").items()]
    numCommunities = max(communities) + 1
    # set colors of vertices
    c = plt.get_cmap("tab20")
    lights = mpl.colors.ListedColormap([c(i)[:3] for i in range(c.N) if  i % 2],"indexed",numCommunities)
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
        
    plt.axis("off")
    #plt.savefig("karate.png")
    plt.show()
    