from utils import pickle_save, pickle_load, validation
from env import environement
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle as cp
def greedy_degree_mvc(graphs):
    objective_vals = []
    for g in graphs:
        degrees = torch.tensor([val for (node, val) in sorted(g.degree(), key=lambda pair: pair[0])])
        env = environement(g)
        Xv , graph = env.reset_env()
        graph = torch.unsqueeze(graph,  0)
        Xv = Xv.clone()
        Xv = Xv.cuda()
        graph = graph.to(device)
        done = False
        non_selected = list(np.arange(env.num_nodes))
        selected = []
        while done == False:
            Xv = Xv.to(device)
            # val = dqn(graph , Xv)[0]
            # val[selected] = -float('inf')
            degrees[selected] = -1
            action = int(torch.argmax(degrees).item())
            Xv_next , reward , done = env.step(action)
            non_selected.remove(action)
            selected.append(action)
            Xv = Xv_next
        objective_vals.append(len(selected))
    # print(objective_vals)
    print(sum(objective_vals)/len(objective_vals))
    return sum(objective_vals)/len(objective_vals)

def read_real_dataset(Path = '../PASDM/code_for_MVC/real_dataset/facebook_combined.txt'):
    edges_unordered = np.genfromtxt("{}".format(Path), dtype=np.int32)
    # nodes
    nodes = set()
    _nodes = edges_unordered.flatten()
    for n in _nodes:
        nodes.add(n)
    # edges
    edges_list = []
    for e in edges_unordered:
        edges_list.append([e[0], e[1]])
    # graph
    g = nx.Graph()
    g.add_nodes_from([i for i in nodes])
    g.add_edges_from(edges_list)

    return g
Path = 'dim128_t10_l2.pkl'
device = 'cuda:0'
# graphs_list = [[read_real_dataset()]]
# graphs_list = pickle_load('../Lab/exp/er_ba_pow_50_100_150_200.pkl')
# graphs_list = pickle_load('../Lab/exp/er_ba_pow_1000.pkl')
# graphs_list = [pickle_load('../PASDM/code_for_MVC/real_dataset/DBLP/BFS_test_n50x50.pkl'),
#                pickle_load('../PASDM/code_for_MVC/real_dataset/DBLP/BFS_test_n100x50.pkl'),
#                pickle_load('../PASDM/code_for_MVC/real_dataset/DBLP/BFS_test_n150x50.pkl'),
#                pickle_load('../PASDM/code_for_MVC/real_dataset/DBLP/BFS_test_n200x50.pkl'),
#                pickle_load('../PASDM/code_for_MVC/real_dataset/DBLP/BFS_test_n1000x50.pkl')]
data_test= '../graph_comb_opt/data/mvc/gtype-erdos_renyi-nrange-500-500-n_graph-1000-p-0.15-m-4.pkl'
f = open(data_test, 'rb')
n_test = 1000
graphs_list = []
for i in tqdm(range(n_test)):
    g = cp.load(f)
    graphs_list.append(g)
graphs_list = [graphs_list]

dqn = torch.load(Path).to(device)
for graphs in graphs_list:
    validation(dqn, graphs)
    # greedy_degree_mvc(graphs)

# graphs_list = []
# test1 = '../PASDM/code_for_MVC/real_dataset/BFS/BFS_test_n150x50.pkl'
# test2 = '../PASDM/code_for_MVC/real_dataset/RW/RW_test_n150x50.pkl'
# test3 = '../PASDM/code_for_MVC/real_dataset/RWRW/RWRW_test_n150x50.pkl'
# test4 = '../PASDM/code_for_MVC/real_dataset/MHRW/MHRW_test_n150x50.pkl'
# graphs_list.append(pickle_load(test1))
# graphs_list.append(pickle_load(test2))
# graphs_list.append(pickle_load(test3))
# graphs_list.append(pickle_load(test4))
# dqns = []
# d1 = 'BFS.pkl'
# d2 = 'RW.pkl'
# d3 = 'RWRW.pkl'
# d4 = 'MHRW.pkl'
# dqns.append(torch.load(d1).to(device))
# dqns.append(torch.load(d2).to(device))
# dqns.append(torch.load(d3).to(device))
# dqns.append(torch.load(d4).to(device))
# for dqn in dqns:
#     print('\n')
#     for graphs in graphs_list:
#         validation(dqn, graphs)

