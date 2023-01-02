import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from utils import training_validation, graph_gen, pickle_save, pickle_load

def main():
    graph_size_list = [50]
    TRAIN_GRAPH_NUM = 1000
    VALIDATION_GRAPH_NUM = 100
    performance_dict = {}
    # print("Running BFS")
    for i in range(len(graph_size_list)):
        num_of_nodes = graph_size_list[i]
        graph_type = 'er'
        p = 0.15
        validation_per_epoch = 10

        train_graphs = graph_gen(n = num_of_nodes, p = p, num = TRAIN_GRAPH_NUM, graph_type = graph_type)
        # train_path = '../PASDM/code_for_MVC/real_dataset/BFS/BFS_train_n50x10000.pkl'
        # train_graphs = pickle_load(train_path)[0:1000]
        val_graphs = graph_gen(n = num_of_nodes, p = p, num = VALIDATION_GRAPH_NUM, graph_type = graph_type)
        
        validation_result, reward_history, loss_history, dqn = training_validation(train_graphs = train_graphs, val_graphs = val_graphs,
                                                                     validation_per_epoch = validation_per_epoch,
                                                                     MAX_EPISODE = TRAIN_GRAPH_NUM,
                                                                     device = 'cuda:0')
        
        legend_word = ['{}'.format(graph_type)]
        plt.plot(validation_result)
        performance_dict[legend_word[0]] = {legend_word[0]:validation_result}
        
        plt.legend(legend_word)
        plt.title("output")
        plt.xlabel("validation")
        plt.ylabel("average vertex picked")
        plt.savefig("remove_edge_val.png")

        plt.clf()
        plt.plot(reward_history)
        plt.savefig("remove_edge_train.png")

        plt.clf()
        plt.plot(loss_history)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.savefig("remove_edge_loss.png")
        #save model for node size 10*i
        Path = 'remove_edge.pkl'
        torch.save(dqn, Path)


if __name__ == '__main__':
    main()