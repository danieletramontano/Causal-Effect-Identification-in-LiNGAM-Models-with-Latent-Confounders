#!/usr/bin/env python3
# -*- codin graph. utf-8 -*-

"FUnctions to  graph.nerate data."
import torch
import networkx as nx
import numpy as np
from random import choice

def hid_var_dag(latent=2,
                observed=5,
                dag_type = 'default',
                parents_max = 3,
                expected_degree=3):

    """
        Generate random Directed Acyclic Graphs (DAGs) with hidden variables.

        Parameters:
        - latent (int): Number of hidden variables.
        - observed (int): Number of observed variables.
        - dag_type (str): Type of DAG to generate ('default' or 'erdos').
        - parents_max (int): Maximum number of parents for observed variables in the 'default' DAG type.
        - expected_degree (int): Expected degree of nodes in the 'erdos' DAG type.

        Returns:
        - g (nx.DiGraph): Generated DAG as a NetworkX DiGraph object.

    """

    adj = np.zeros((latent+observed, latent+observed))
    for i in range(latent):
        obs_ch = np.random.randint(latent,latent+observed,np.random.randint(2,observed+1))
        for j in obs_ch:
            adj[i,j] = 1

    if dag_type == 'default':
        for j in range(1, observed):
            nb_parents = np.random.randint(0, min([parents_max, j])+1)
            for i in np.random.choice(range(0, j), nb_parents, replace=False):
                adj[latent+i,latent+j] = 1

    elif dag_type == 'erdos':
        nb_edges = expected_degree * observed
        prob_connection = 2 * nb_edges/(observed**2 - observed)
        causal_order = np.random.permutation(np.arange(observed))

        for i in range(observed - 1):
            node = causal_order[i]
            possible_parents = causal_order[(i+1):]
            num_parents = np.random.binomial(n=observed - i - 1,
                                             p=prob_connection)
            parents = np.random.choice(possible_parents, size=num_parents,
                                       replace=False)
            adj[latent+parents,latent+node] = 1

    graph = nx.DiGraph(adj)
    return graph





"GENERATE PARAMETERS"
def par_gen(latent=2,
            observed=5,
            graph = None,
            graph_adj = None,
            same_var = True,
            v_min = 0.2,
            v_max = 10,
            stn_ratio = 1,
            w_max = 10):

    varr = par_gen_var(latent=latent,
                       observed=observed,
                       graph =  graph,
                       graph_adj =  graph_adj,
                       same_var = same_var,
                       v_min = v_min,
                       v_max = v_max,
                       stn_ratio = stn_ratio)

    weight, B = par_gen_weights(graph =  graph,
                                w_max = w_max)

    return varr, weight, B

def par_gen_var(latent=2,
                observed=5,
                graph = None,
                graph_adj = None,
                same_var = True,
                v_min = 0.2,
                v_max = 10,
                stn_ratio = 1):

    varr = torch.Tensor(len(graph.nodes()))
    if same_var == 'True':
        varr[range(latent)] = torch.ones(observed)
        varr[range(latent,latent+observed)] = torch.ones(observed)

    else:
        varr[range(latent)] = torch.Tensor(latent).uniform_(v_min,v_max)
        varr[range(latent, latent+observed)] = torch.Tensor(observed).uniform_(v_min,v_max)
    
    if stn_ratio > 0:
        for i in range(latent):
            j = latent
            find_first_ch = False
            while(j < latent + observed and find_first_ch is False):
                if (graph_adj[i, j] == 1):
                    find_first_ch = True
                j += 1
            if find_first_ch is False:
                varr[i] = torch.Tensor(1).uniform_(v_min,v_max)
            else:
                varr[i] = varr[j]*stn_ratio

    return varr

def par_gen_weights(graph = None,
                    w_max = 10,
                    no_small = True
                    ):
    
    if no_small is True:
        weight = torch.Tensor(len(graph.edges())).uniform_(0.2,w_max) 
        weight = weight * torch.Tensor([choice([-1, 1]) for i in range(len(graph.edges()))])
    else: 
        weight = torch.Tensor(len(graph.edges())).uniform_(-w_max,w_max)

    adj = torch.eye(len(graph.nodes()))
    for e in range(len(graph.edges())):
        adj[list(graph.edges)[e]]=-weight[e]
    B = (torch.inverse(adj)).t()

    return weight, B




def hid_var_data(latent=2,
                 observed=5,
                 n = 500,
                 B = None,
                 distr = 'Laplace',
                 varr = torch.ones(7)
                 ):
    """
    Generate, possibly whitened, data from a given DAG.

    Parameters:
        - latent (int): Number of hidden variables.
        - observed (int): Number of observed variables.
        - g (nx.DiGraph): The DAG as a NetworkX DiGraph object.
        - n (int): Number of data samples to generate.
        - withening (bool): Flag to apply whitening to the data.
        - distr(string): 'Laplace' or 'Cauchy'
        - same_var(bool): True if the noise terms must have the same variance

    Returns:
        - data (torch.Tensor): Generated data.
        - adj (torch.Tensor): Adjacency matrix of the DAG.
        - W_with (torch.Tensor): Whitening matrix (if applicable otherwise identity matrix).
        - weight (torch.Tensor): Weights associated with the DAG edges.

    """

    err = torch.Tensor(latent+observed,n)

    for j in range(latent+observed):
        for i in range(n):
            if distr == 'Laplace':
                err[j,i] = torch.distributions.laplace.Laplace(0,1).sample()*varr[j]
            elif distr == 'Cauchy':
                err[j, i] = torch.Tensor(1).cauchy_()*varr[j]
            elif distr == 'Normal':
                err[j, i] = torch.Tensor(1).normal_()*varr[j]
            else:
                print(distr , 'error')
                break



    #err = torch.Tensor(latent+observed,n).cauchy_()
    data = B.matmul(err).t()
    data = data[:,range(latent, observed+latent)]

    return data
