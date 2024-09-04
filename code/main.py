import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import CommunityDiscovery,Helper,Projection,Similarity

## load edgelist and attributes as pandas dataframes
df = pd.read_csv("../data/disease_edgelist.csv")
df_att = pd.read_csv("../data/disease_attributes.csv",index_col="Id")

## create bipartite graph from dataframe and assign attributes
G = nx.convert_matrix.from_pandas_edgelist(df,"Source","Target",create_using=nx.Graph())
nx.set_node_attributes(G,df_att.to_dict(orient="index"))

## get all our functions
proj = Projection()
cd = CommunityDiscovery()
util = Helper()
sim = Similarity()

## make projections
proj_names = ["simple","unweight","ycn"]

hdn_simple,gdn_simple = proj.simple_weight(G,["Subclass"])
hdn_unw,gdn_unw = proj.unweight(G,["Subclass"])
hdn_ycn,gdn_ycn = proj.ycn(G,["Subclass"])

## make community discovery
com_names = ["infomap","greedy"]

# set some parameters for the community discovery (infomap)
im_params = ["--two-level",
             "--num-trials 10",
             "--teleportation-probability 0.15",
             "--markov-time 2",
             "--core-loop-limit 10",
             "--tune-iteration-threshold 1e-5"]

for com_name in com_names:
    # simple weight
    hdn_simple,simple_info_coms = cd.infomap_detect(hdn_simple,parameters=im_params)
    hdn_simple,simple_greedy_coms = cd.greedy(hdn_simple)

    # unweight
    hdn_unw,unw_info_coms = cd.infomap_detect(hdn_unw,parameters=im_params)
    hdn_unw,unw_greedy_coms = cd.greedy(hdn_unw)

    # ycn
    hdn_ycn,ycn_info_coms = cd.infomap_detect(hdn_ycn,parameters=im_params)
    hdn_ycn,ycn_greedy_coms = cd.greedy(hdn_ycn)

graphs = [hdn_simple,hdn_unw,hdn_ycn]

communities = [simple_info_coms,simple_greedy_coms,
               unw_info_coms,unw_greedy_coms,
               ycn_info_coms,ycn_greedy_coms]

## normalized mutual information
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
nmis = np.ones((6,6))
for idx1,c1 in enumerate(communities):
    for idx2,c2 in enumerate(communities):
        if idx2 == idx1:
            continue
        elif idx2 < idx1:
            nmis[idx1,idx2] = sim.norm_mutual_info(c2,c1)
            continue
        proj_name1 = proj_names[int(idx1/2)]
        com_name1 = com_names[int(idx1%2)]
        proj_name2 = proj_names[int(idx2/2)]
        com_name2 = com_names[int(idx2%2)]
        nmi = sim.norm_mutual_info(c1,c2)
        nmis[idx1,idx2] = nmi
        print("NMI for {}_{} and {}_{} is {:.3f}".format(proj_name1,com_name1,
                                                         proj_name2,com_name2,
                                                         nmi))

ax.imshow(nmis,vmin=0.5,cmap="magma_r")
ax.set_title("Normalized mutual information between pairs of\nprojection and community discovery algorithms")
plt.show()
#annotate_heatmap(im, valfmt="{x:d}", size=7, threshold=20,
#                 textcolors=["red", "white"])

## draw the communities
fig = plt.figure(figsize=(16,24))

i = 0
for graph in graphs:
    for com_name in com_names:
        ax = fig.add_subplot(3,2,i+1)
        util.drawCommunities(graph,ax,com_name,proj_names[int(i/2)])
        i += 1

plt.show()
