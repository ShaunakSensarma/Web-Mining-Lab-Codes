import matplotlib.pyplot as plt 
import networkx as nx 
  
G = nx.karate_club_graph() 
  
plt.figure(figsize =(15, 15)) 
nx.draw_networkx(G, with_labels = True) 

Gl = nx.read_edgelist("wiki-Vote.txt.gz", create_using = nx.DiGraph(Directed=True), nodetype=int)
print(nx.info(Gl))
nx.draw_networkx(Gl, with_labels = True) 

n_nodes=10000

#degree_prestige = dict((v,len(Gl.in_edges(v))/(n_nodes-1)) for v in Gl.nodes())
#print("\\n\nDEGREE PRESTIGE :\n")

#for i in degree_prestige:
#    print(i, " : ", degree_prestige[i])
print("\n\n\n\n\n\n\n")

#distance = []
#temp_dis = 0
#n = 0
#for dest in Gl:
#    temp_dis = 0
#    n = 0
#    for src in Gl:
#       if (nx.has_path(Gl,src,dest) == True):
#            temp_dis = temp_dis + nx.shortest_path_length(Gl,source = src,target = dest)
#            n = n + 1
#    if temp_dis == 0:
#        distance.append([dest, 0])
#    else:
#        distance.append([dest, temp_dis/(n - 1)])
#print("\nPROXIMITY PRESTIGE :\n")
#a=0
#for i in distance:
#    print(str(i[0]) + " : " + str(i[1]))
#    a=a+1
#    if(a==500):
#        break
            


rank_prestige = nx.zeros([n_nodes], dtype = int)

path_matrix = nx.zeros([n_nodes, n_nodes], dtype = int)
i = 0
j = 0
for src in Gl:
    for dest in Gl:
        if Gl.has_edge(dest, src):
            path_matrix[i][j] = 1
        j = j+1
    j = 0
    i = i+1
for i in range(n_nodes):
    pr_i = 0
    for j in range(n_nodes):
        pr_i = pr_i + path_matrix[i][j] * Gl[j]
    rank_prestige[i] = pr_i

print("\nRANK PRESTIGE :\n")
print(rank_prestige)

