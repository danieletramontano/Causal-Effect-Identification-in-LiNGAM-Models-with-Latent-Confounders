import torch
import numpy as np
import networkx as nx


def generate_data(g, observed, latent, weights=None, mode="fixed", 
                       n_samples=1000, distr_name="Laplace",
                       distr_params_default=True, distr_params=None):
    """
    Generate data from a given DAG.
    
    Parameters:
        - g (nx.DiGraph): The DAG as a NetworkX DiGraph object.
        - observed (int): Number of observed variables.
        - latent (int): Number of hidden variables.
        - weights (torch.Tensor): Weights associated with the DAG edges (could be None).
        - mode (string): if weights is None, defines how the weights are generated.
        - n_samples (int): Number of data samples to generate.
        - distr_name (string): ['Laplace', 'Cauchy', 'Exponential']
        - distr_params_default (bool): True if no specific parameters for exogenous noises are specified
        - distr_params (dict): None or dict with specific parameters for exogenous noises
    
    Returns:
        - data (torch.Tensor): Generated data.
        - adj (torch.Tensor): Adjacency matrix of the DAG.
        - weights (torch.Tensor): Weights associated with the DAG edges.
    """
    n_weights = len(g.edges())
    if weights is None:
        if mode=="fixed":
            weights = torch.full((n_weights,), 0.9)
        elif mode=="random":
            weights = torch.Tensor(n_weights).uniform_(-0.5, 0.5)
            for i in range(n_weights):
                if weights[i]>0:
                    weights[i] += 0.5
                else:
                    weights[i] -= 0.5
    
    adj = torch.eye(len(g.nodes()))
    for e in range(len(g.edges())):
        adj[list(g.edges)[e]]=-weights[e]
    B = (torch.inverse(adj)).t()

    if distr_name=="Laplace":
        distr = torch.distributions.laplace.Laplace
        if distr_params_default:
            distr_params = [{"loc": 0, "scale": 1}] * (latent+observed)
    elif distr_name=="Cauchy":
        distr = torch.distributions.cauchy.Cauchy
        if distr_params_default:
            distr_params = [{"loc": 0, "scale": 1}] * (latent+observed)
    elif distr_name=="Exponential":
        distr = torch.distributions.exponential.Exponential
        if distr_params_default:
            distr_params = [{"rate": 1}] * (latent+observed)
    
    err = torch.Tensor(latent+observed, n_samples)
    for j in range(latent+observed):
        tmp  = distr(**distr_params[j]).sample((n_samples,))
        tmp  -= tmp.mean()
        err[j, :] = tmp
                
    data = B.matmul(err).t()
    data = data[:,range(latent, observed+latent)]
    
    return data, adj, weights


def generate_data_misspecification(g, observed, latent, weights_lin=dict(), 
                                   weights=None, mode="fixed",
                                   n_samples=1000, transform=torch.tanh,
                                   distr_name='Laplace', distr_params_default=True,
                                   distr_params=None):
    """
    Generate data from a given DAG with misspecification.
    
    Parameters:
        - g (nx.DiGraph): The DAG as a NetworkX DiGraph object.
        - observed (int): Number of observed variables.
        - latent (int): Number of hidden variables.
        - weights_lin (dcit): Dict of linear transformations (that are not misspecified).
        - weights (torch.Tensor): Weights associated with the DAG edges (could be None).
        - mode (string): if weights is None, defines how the weights are generated.
        - n_samples (int): Number of data samples to generate.
        - transform (func): Non-linear trunsformation applied as misspecification
        - distr_name (string): ['Laplace', 'Cauchy', 'Exponential']
        - distr_params_default (bool): True if no specific parameters for exogenous noises are specified
        - distr_params (dict): None or dict with specific parameters for exogenous noises
    
    Returns:
        - data (torch.Tensor): Generated data.
        - adj (torch.Tensor): Adjacency matrix of the DAG.
        - weights (torch.Tensor): Weights associated with the DAG edges.
    
    """
    n_weights = len(g.edges())
    if weights is None:
        if mode=="fixed":
            weights = torch.full((n_weights,), 0.9)
        elif mode=="random":
            weights = torch.Tensor(n_weights).uniform_(-0.5, 0.5)
            for i in range(n_weights):
                if weights[i]>0:
                    weights[i] += 0.5
                else:
                    weights[i] -= 0.5
    
    n_nodes = len(g.nodes())
    adj = torch.eye(n_nodes)
    for e in range(n_nodes):
        adj[list(g.edges)[e]]=-weights[e]
    B = (torch.inverse(adj)).t()
    
    err = torch.Tensor(latent+observed,n_samples)
    if distr_name=="Laplace":
        distr = torch.distributions.laplace.Laplace
        if distr_params_default:
            distr_params = [{"loc": 0, "scale": 1}] * (latent+observed)
    elif distr_name=="Cauchy":
        distr = torch.distributions.cauchy.Cauchy
        if distr_params_default:
            distr_params = [{"loc": 0, "scale": 1}] * (latent+observed)
    elif distr_name=="Exponential":
        distr = torch.distributions.exponential.Exponential
        if distr_params_default:
            distr_params = [{"rate": 1}] * (latent+observed)
    
    for j in range(latent+observed):
        tmp  = distr(**distr_params[j]).sample((n_samples,))
        tmp  -= tmp.mean()
        err[j, :] = tmp

    ordered_nodes = []
    visited_nodes = []
    nodes_set = set(g.nodes())
    node = None
    while len(nodes_set)>0:
        if node==None:
            if len(visited_nodes) > 0:
                node = visited_nodes[-1]
            else:
                node = next(iter(nodes_set))
                visited_nodes.append(node)
        parents = set(g.predecessors(node))
        if parents.issubset(set(ordered_nodes)):
            ordered_nodes.append(node)
            nodes_set.remove(node)
            visited_nodes.pop(-1)
            node = None
        else:
            tmp_set = set(ordered_nodes) - set(parents)
            node = next(iter(tmp_set))
            visited_nodes.append(node)         

    A = torch.zeros((n_nodes, n_nodes))
    for i, e in enumerate(g.edges):
        A[e] = weights[i]
    
    data = torch.zeros((n_samples, latent+observed))
    for col in ordered_nodes:
        if col in weights_lin.keys():
            data_lin_part = torch.zeros((n_samples,))
            for edge in weights_lin[col]:
                data_lin_part += data[:, edge[0]]*A[edge]
            data[:, col] = transform(torch.matmul(data, A[col]) - data_lin_part) + data_lin_part + err[col, :]
        else:
            data[:, col] = torch.matmul(data, A[col]) + err[col, :]
    
    data = data[:,range(latent, latent+observed)]
    
    return data, adj, weights


if __name__=="main":
    observed = 3
    latent = 1
    iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    g = nx.DiGraph(iv_adj)
    weights = None
    mode = "fixed"
    n_samples = 1000
    distr_name = "Laplace"
    distr_params_default = True
    distr_params = None
    data, adj, weights = generate_data(
        g=g, observed=observed, latent=latent, weights=weights, 
        mode=mode, n_samples=n_samples, distr_name=distr_name, 
        distr_params_default=distr_params_default, distr_params=distr_params)