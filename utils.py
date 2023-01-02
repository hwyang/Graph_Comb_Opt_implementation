import numpy as np
from collections import namedtuple
from env import environement
from DQN_network import embedding_network
import networkx as nx
import math
import random
import torch
from tqdm import tqdm
import pickle
from heapdict import heapdict
from networkx.algorithms.approximation import min_weighted_vertex_cover
import itertools
from itertools import cycle
import warnings
import torch.multiprocessing as mp
import copy
warnings.filterwarnings("ignore")

# experience = namedtuple("experience" , ['graph','Xv','action','reward','next_Xv','is_done'])
experience = namedtuple("experience" , ['graph','Xv','action','reward','next_graph','next_Xv','is_done'])

def pickle_save(data,file_name):
    with open(file_name,'wb') as f:
        pickle.dump(data , f)
def pickle_load(file_name):
    with open(file_name,'rb') as f:
        return pickle.load(f)

class replay_buffer():
    def __init__(self , max_size):
        self.buffer = np.zeros([max_size], dtype = experience)
        self.max_size = max_size
        self.size = 0
        self.idx = -1

    def push(self, new_exp):
        if(self.size >= self.max_size):
            self.idx = (self.idx+1) % self.max_size
        else:
            self.idx = self.idx + 1
            self.size += 1

        self.buffer[self.idx] = new_exp
    
    def sample(self, batch_size):
        batch = np.random.choice(np.arange(self.size) , size = batch_size , replace=False)
        return self.buffer[[batch]]

    def clear_buffer(self):
        self.size = 0
        self.idx = -1  

#validation the result
def validation(dqn , validation_graph , device = 'cuda:0'):
    objective_vals = []
    for g in validation_graph:
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
            #Xv = Xv.cuda()
            Xv = Xv.to(device)
            val = dqn(graph , Xv)[0]
            val[selected] = -float('inf')
            action = int(torch.argmax(val).item())
            Xv_next , reward , done = env.step(action)
            non_selected.remove(action)
            selected.append(action)
            Xv = Xv_next
        objective_vals.append(len(selected))
    # print(objective_vals)
    print(sum(objective_vals)/len(objective_vals))
    return sum(objective_vals)/len(objective_vals)

def training_validation(train_graphs, val_graphs, validation_per_epoch, MAX_EPISODE, device = 'cuda:0'):
    # define hyper paramters and initial dqn
    torch.cuda.manual_seed_all(19960214)
    torch.manual_seed(19960214)
    np.random.seed(19960214)
    random.seed(19960214)

    EPS_START = 1.00
    EPS_END = 0.05
    EPS_DECAY = 3000 #15000
    emb_dim = 64
    T = 5
    dqn = embedding_network(emb_dim = emb_dim, T = T, device = device, init_factor = 10, w_scale = 0.01).double()
    target_net = embedding_network(emb_dim = emb_dim, T = T, device = device, init_factor = 10, w_scale = 0.01).double()
    target_net.load_state_dict(dqn.state_dict())
    ## For Multiprocessing
    # mp.set_start_method('spawn')
    # dqn.share_memory()

    steps_done = 0
    loss_func = torch.nn.MSELoss()
    USE_CUDA = torch.cuda.is_available()
    N_STEP = 2
    
    reward_history = []
    loss_history = []
    validation_result = []
    batch_size = 64
    buffer = replay_buffer(500000)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=0.0001, amsgrad=False)
    
    fitted_q_exp = namedtuple("fitted_exp" , ['graph','Xv','action','reward'])

    if USE_CUDA:
        dqn.to(device)
        target_net.to(device)
    
    # main training loop
    #for e in tqdm(range(MAX_EPISODE)):
    #    g = train_graphs[e]
    graph_pool = cycle(train_graphs)
    for e in tqdm(range(3000)):
        g = next(graph_pool)
        # print(g.number_of_edges())
        env = environement(g)
        Xv, graph = env.reset_env()
        graph = torch.unsqueeze(graph, 0)
        graph = graph.clone()
        Xv = Xv.clone()
        
        
        done = False
        non_selected = list(np.arange(env.num_nodes))
        selected = []
        eps_reward = []

        N = 0
        reward_list = []
        fitted_experience_list = []
        #cur_episode_loss = []

        if USE_CUDA:
            graph = graph.to(device)
            Xv = Xv.to(device)
        
        while done == False:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            #eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)
            
            if USE_CUDA:
                graph = graph.to(device)
                Xv = Xv.to(device)

            if np.random.uniform() > eps_threshold:
                val = dqn(graph , Xv)[0]
                val[selected] = -float('inf')
                # print(val)
                action = int(torch.argmax(val).item())
                # action = list(env.nx_graph.nodes)[action]
            else:
                action = int(np.random.choice(non_selected))
                # action = int(np.random.choice(list(env.nx_graph.nodes)))
            # index = list(env.nx_graph.nodes).index(action)
            # Xv_next, reward , done = env.step(action)
            # graph_next, Xv_next, reward, done = env.graph_remove_selected_node_step(action)
            graph_next, Xv_next, reward, done = env.graph_remove_edge_step(action)
            graph_next = torch.unsqueeze(graph_next, 0)
            graph_next = graph_next.clone()

            eps_reward.append(reward)
            Xv_next = Xv_next.clone()
            
            fit_ex = fitted_q_exp(graph , Xv , action , reward)
            # fit_ex = fitted_q_exp(graph , Xv , index , reward)
            fitted_experience_list.append(fit_ex)
            
            non_selected.remove(action)
            selected.append(action)
            reward_list.append(reward)
            
            N = N + 1
            if N >= N_STEP:
                n_prev_ex = fitted_experience_list[0]
                n_graph = n_prev_ex.graph
                n_Xv = n_prev_ex.Xv
                n_action = n_prev_ex.action
                n_reward = sum(reward_list)
                # ex = experience(n_graph , n_Xv , torch.tensor([n_action]) , torch.tensor([n_reward]) , Xv_next , done)
                ex = experience(n_graph, n_Xv, torch.tensor([n_action]), torch.tensor([n_reward]), graph_next, Xv_next, done)
                buffer.push(ex)
                fitted_experience_list.pop(0)
                reward_list.pop(0)
            #ex = experience(graph , Xv , torch.tensor([action]) , torch.tensor([reward]) , Xv_next , done)
            #buffer.push(ex)
            Xv = Xv_next
            graph = graph_next
            steps_done += 1

            # num_of_process = batch_size
            # processes = []
            # if buffer.size >= batch_size:
            #     for rank in range(num_of_process):
            #         p = mp.Process(target=Parallel_Train, args=(rank, dqn, target_net, buffer, USE_CUDA, device, loss_func, optimizer))
            #         p.start()
            #         processes.append(p)
            #     for p in processes:
            #         p.terminate()
            #         p.join()
            if buffer.size >= batch_size:
                # loss = 0
                # sample_size = 1
                # for _ in range(batch_size):
                # Sample = buffer.sample(sample_size)
                Sample = buffer.sample(batch_size)
                batch = experience(*zip(*Sample))
                # batch_graph, batch_state, batch_action, batch_reward, batch_next_state = \
                #     tuple(map(torch.cat, (batch.graph, batch.Xv, batch.action, batch.reward, batch.next_Xv)))
                batch_graph, batch_state, batch_action, batch_reward, batch_next_graph, batch_next_state = \
                    tuple(map(torch.cat, (batch.graph, batch.Xv, batch.action, batch.reward, batch.next_graph, batch.next_Xv)))
                '''
                batch_graph = torch.cat(batch.graph)
                batch_state = torch.cat(batch.Xv)
                batch_action = torch.cat(batch.action)
                batch_reward = torch.cat(batch.reward).double()
                batch_next_state = torch.cat(batch.next_Xv)
                '''
                non_final_mask = torch.tensor(tuple(map(lambda s : s is not True, batch.is_done)), dtype = torch.uint8).detach()
                # non_final_graph = batch_graph[non_final_mask].detach()
                non_final_graph = batch_next_graph[non_final_mask].detach()
                non_final_next_state = batch_next_state[non_final_mask].detach()

                # next_state_value = torch.zeros(sample_size).detach().double().detach()
                next_state_value = torch.zeros(batch_size).detach().double().detach()
                if USE_CUDA:
                    batch_graph = batch_graph.to(device)
                    batch_state = batch_state.to(device)
                    batch_action = batch_action.to(device)
                    batch_reward = batch_reward.to(device)
                    batch_next_graph = batch_next_graph.to(device)
                    batch_next_state = batch_next_state.to(device)
                    next_state_value = next_state_value.to(device)
                    non_final_graph = non_final_graph.to(device)
                    non_final_next_state = non_final_next_state.to(device)

                # next_state_value[non_final_mask] = target_net(non_final_graph, non_final_next_state).max(1)[0].detach()
                if batch.is_done == False:
                    next_state_value[non_final_mask] = target_net(non_final_graph, non_final_next_state).max(1)[0].detach()
                expected_state_action_values = next_state_value + batch_reward
                dqn_out = dqn(batch_graph, batch_state) #
                state_action_values = dqn_out.gather(1 , batch_action.view(-1,1)).view(-1)
                # loss += loss_func(state_action_values, expected_state_action_values)
                ##
                # l2_loss = 0.0
                # for b in range(batch_size):
                #     for q in dqn_out[b]:
                #         l2_loss += (state_action_values[b] - q) * (state_action_values[b] - q)
                ##
                loss = loss_func(state_action_values, expected_state_action_values)
                # loss += l2_loss #
                loss_history.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if e > 0 and e % 10 == 0:
            target_net.load_state_dict(dqn.state_dict())
        if e > 0 and e % validation_per_epoch == 0:
            v = validation(dqn, val_graphs, device=device)
            validation_result.append(v)  
        reward_history.append(sum(eps_reward))
        # print(sum(eps_reward))
    return validation_result, reward_history, loss_history, dqn

def graph_gen(n , p , num, graph_type = 'er'):
    np.random.seed(19960214)
    random.seed(19960214)
    validation_graph = []
    if graph_type == 'er':
        for i in range(num):
            p = random.uniform(0.15 , 0.15)
            # n = random.randint(50, 100)
            g = nx.erdos_renyi_graph(n , p)
            validation_graph.append(g)
    elif graph_type == 'ba':
        for i in range(num):
            m = density_to_edge_ba(n , p)
            g = nx.barabasi_albert_graph(n , m)
            validation_graph.append(g)
    elif graph_type == 'reg':
        for i in range(num):
            d = density2regD(n , p)
            g = nx.random_regular_graph(n = n , d = d)
            validation_graph.append(g)
    elif graph_type == 'pow':
        for i in range(num):
            m = density_to_edge_ba(n , p)
            g = nx.powerlaw_cluster_graph(n = n , m = m , p = 0.25)
            validation_graph.append(g)
    else:
        g = nx.Graph()
        for i in range(n):
            g.add_node(i)
        for i in range(num):
            e = random.choice(list(nx.non_edges(g)))
            g.add_edge(*e)
            g = copy.deepcopy(g)
            validation_graph.append(g)
    return validation_graph


# def Parallel_Train(rank, dqn, target_net, buffer, USE_CUDA, device, loss_func, optimizer):
#     torch.cuda.manual_seed_all(19960214+rank)
#     torch.manual_seed(19960214+rank)
#     np.random.seed(19960214+rank)
#     random.seed(19960214+rank)

#     sample_size = 1
#     Sample = buffer.sample(sample_size)
#     batch = experience(*zip(*Sample))
#     batch_graph = torch.cat(batch.graph)
#     batch_state = torch.cat(batch.Xv)
#     batch_action = torch.cat(batch.action)
#     batch_reward = torch.cat(batch.reward).double()
#     batch_next_state = torch.cat(batch.next_Xv)

#     non_final_mask = torch.tensor(tuple(map(lambda s : s is not True, batch.is_done)), dtype = torch.uint8).detach()
#     non_final_graph = batch_graph[non_final_mask].detach()
#     non_final_next_state = batch_next_state[non_final_mask].detach()

#     next_state_value = torch.zeros(sample_size).detach().double().detach()
#     if USE_CUDA:
#         batch_graph = batch_graph.to(device)
#         batch_state = batch_state.to(device)
#         batch_action = batch_action.to(device)
#         batch_reward = batch_reward.to(device)
#         batch_next_state = batch_next_state.to(device)
#         next_state_value = next_state_value.to(device)
#         non_final_graph = non_final_graph.to(device)
#         non_final_next_state = non_final_next_state.to(device)

#     next_state_value[non_final_mask] = target_net(non_final_graph, non_final_next_state).max(1)[0].detach()

#     state_action_values = dqn(batch_graph, batch_state).gather(1 , batch_action.view(-1,1)).view(-1)
#     expected_state_action_values = next_state_value + batch_reward

#     loss = loss_func(state_action_values, expected_state_action_values)
#     print(f'Rank:{rank}, Loss:{loss}')
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()