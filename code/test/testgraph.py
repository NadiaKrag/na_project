from projection import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from network_map import ycn_edges

"""
import pandas as pd
from time import time
df = pd.read_csv("../../data/disease_edgelist.csv")
df_att = pd.read_csv("../../data/disease_attributes.csv",index_col="Id")
G = nx.convert_matrix.from_pandas_edgelist(df,"Source","Target",create_using=nx.Graph())
nx.set_node_attributes(G,df_att.to_dict(orient="index"))

t1 = time()
G_dict = nx.to_dict_of_dicts(G)
projs1,projs2 = {n:(1) if n in df.Source.tolist() else (0) for n in G.nodes},{n:(0) if n in df.Source.tolist() else (1) for n in G.nodes}
test1,test2 = ycn_edges(G_dict,projs1),ycn_edges(G_dict,projs2)
print(test1[(3985, 3986)],test1[(3796, 3799)],test1[(3471, 3484)],test1[(1625, 1629)])
print(time()-t1)

t1 = time()
U,V = ycn(G)
test = nx.get_edge_attributes(V,"weight")
print(test[(3985, 3986)],test[(3796, 3799)],test[(3471, 3484)],test[(1625, 1629)])
print(time()-t1)
exit()
"""

# bipartite graph with odd vertices on the left and even on the right
G = nx.Graph()
edge_list = [(1,0),
             (1,2),
             (1,4),
             (1,6),
             (1,8),
             (3,0),
             (5,2),
             (5,4),
             (5,6),
             (7,4),
             (7,6),
             (9,4),
             (9,6)]

G.add_edges_from(edge_list)
nx.set_node_attributes(G,{i:({"parity":"odd"} if i % 2 else {"parity":"even"}) for i in range(10)})

U,V = ycn(G)

G_dict = nx.to_dict_of_dicts(G)
projs = {n:(1) if n % 2 == 1 else (0) for n in G.nodes}
edges = ycn_edges(G_dict,projs)

fig = plt.figure(figsize=(21,7))

# position vertices on left and right dependent on bipartite set
ax_G = fig.add_subplot(131)
G_pos = {n:(1,i) if i % 2 else (2,i+1) for i,n in enumerate(sorted(G.nodes))}
nx.draw_networkx(G,pos=G_pos,with_labels=True,ax=ax_G)

# position in a circle
ax_U = fig.add_subplot(132)
U_pos = {n:(np.cos((n/len(U))*2*np.pi),np.sin((n/len(U))*2*np.pi)) for n in U.nodes()}
nx.draw_networkx(U,pos=U_pos,with_labels=True,ax=ax_U)
nx.draw_networkx_edge_labels(U,pos=U_pos,ax=ax_U)

ax_V = fig.add_subplot(133)
V_pos = {n:(np.cos((n/len(V))*2*np.pi),np.sin((n/len(V))*2*np.pi)) for n in V.nodes()}
nx.draw_networkx(V,pos=V_pos,with_labels=True,ax=ax_V)
nx.draw_networkx_edge_labels(V,pos=V_pos,ax=ax_V)

plt.show()
