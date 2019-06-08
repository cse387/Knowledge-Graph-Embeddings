import pylab
from matplotlib import pyplot as plt
import networkx as nx
def Visualize_querytype2(query,aproximation,groud_truth):
    G=nx.DiGraph()
    pos={query[0][0]:[0.1,0.7],query[1][0]:[0.1,0.5],query[0][1]:[0.35,0.6],'?':[0.6,0.6]}
    plt.title('Query Structure')
    G.add_edges_from(query)

    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 1000)
    nx.draw_networkx_edges(G, pos, edge_color='b', arrows=True, arrowsize=30)
    nx.draw_networkx_labels(G, pos)
    plt.text(0.09,0.73,'Gene Names',size=10,bbox=dict(boxstyle="square",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       ))
    plt.text(0.34,0.73,'Tissue',size=10,bbox=dict(boxstyle="square",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       ))
    plt.figure()
    Ground_Truth=nx.Graph()
    Ground_Truth.add_nodes_from(groud_truth)

    pos_Ground_Truth={}
    init_pos1=[0.3,0.5]
    for node in Ground_Truth.nodes:
        temp=init_pos1[:]
        if node not in pos_Ground_Truth.keys():
            pos_Ground_Truth[node]=temp
        init_pos1[1]-=0.2
        

    Nearest_Answers=nx.Graph()
    Nearest_Answers.add_nodes_from(aproximation)

    pos_Nearest_Answers={}
    init_pos2=[0.43,0.5]
    for node in Nearest_Answers.nodes:
        temp=init_pos2[:]
        if node not in pos_Nearest_Answers.keys():
            pos_Nearest_Answers[node]=temp
        init_pos2[1]-=0.2
        
    nx.draw_networkx_nodes(Ground_Truth,pos_Ground_Truth,cmap=plt.get_cmap('jet'), node_size = 1000)
    nx.draw_networkx_labels(Ground_Truth, pos_Ground_Truth)
    nx.draw_networkx_nodes(Nearest_Answers,pos_Nearest_Answers,cmap=plt.get_cmap('jet'), node_size = 1000)
    nx.draw_networkx_labels(Nearest_Answers, pos_Nearest_Answers)
    plt.text(0.29,0.6,'Ground Truth',size=10,bbox=dict(boxstyle="square",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       ))
    plt.text(0.42,0.6,'Nearest Answers',size=10,bbox=dict(boxstyle="square",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       ))
    for ground_key in pos_Ground_Truth.keys():
        for approximate_key in pos_Nearest_Answers:
            if ground_key==approximate_key:
                x=[pos_Ground_Truth[ground_key][0],pos_Nearest_Answers[approximate_key][0]]
                y=[pos_Ground_Truth[ground_key][1],pos_Nearest_Answers[approximate_key][1]]
                plt.plot(x,y,color='green', linestyle='dashed')

    pylab.show()

