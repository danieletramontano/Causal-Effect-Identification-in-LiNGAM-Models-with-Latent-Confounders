import math
import torch
import numpy as np
import networkx as nx

from sklearn.linear_model import LinearRegression
from .utils import intersection, count_lists
from torch.nn import Parameter


def get_ratio_(Z, D, deg=2):
    """
    Computes ratio between alpha_d/alpha_z

    Parameters:
        - Z (np.array): proxy variable observations
        - D (np.array): treatment observations
        - deg (int): moment of non-guassianity (equal to the (n-1) from the original paper)
    """
    var_u = np.mean(Z*D)
    sign = np.sign(var_u)
    
    diff_normal_D = np.mean(D**(deg)*Z) - deg*var_u*np.mean(D**(deg-1))
    diff_normal_Z = np.mean(Z**(deg)*D) - deg*var_u*np.mean(Z**(deg-1))
    
    alpha_sq = ((diff_normal_D) / (diff_normal_Z))
    if alpha_sq < 0:
        alpha_sq = -(abs(alpha_sq)**(1/(deg-1)))
    else:
        alpha_sq = alpha_sq**(1/(deg-1))
    alpha_sq = abs(alpha_sq)*sign
    
    return alpha_sq


def cross_moment(Z, D, Y, deg=2):
    """
    Cross-Moment method implementation

    Parameters:
        - Z (np.array): proxy variable observations
        - D (np.array): treatment observations
        - Y (np.array): outcome observations
        - deg (int): moment of non-guassianity (equal to the (n-1) from the original paper)
    """
    denominator = 0
    while denominator==0:
        alpha_sq = get_ratio_(Z, D, deg)
        numerator = np.mean(D*Y) - alpha_sq*np.mean(Y*Z)
        denominator = np.mean(D*D) - alpha_sq*np.mean(D*Z)
        deg += 1
    return numerator / denominator


def cross_moment_sensor_fusion(W, Z, D, Y, deg=2, n_iter=100, percentage=0.9):
    """
    Cross-Moment method with two proxies

    Parameters:
        - W (np.array): first proxy variable observations
        - Z (np.array): second proxy variable observations
        - D (np.array): treatment observations
        - Y (np.array): outcome observations
        - deg (int): moment of non-guassianity (equal to the (n-1) from the original paper)
        - n_iter: number of bootstrap samplings
        - 
    """
    length = len(W)
    sample_set = np.arange(length)
    n_samples = math.ceil(percentage*length)
    betas_est1 = np.zeros(n_iter)
    betas_est2 = np.zeros(n_iter)
    for it in range(n_iter):
        args_id = np.random.choice(sample_set, size=n_samples)
        W_tmp = W[args_id]
        Z_tmp = Z[args_id]
        D_tmp = D[args_id]
        Y_tmp = Y[args_id]
        
        beta_est1 = cross_moment(Z_tmp, D_tmp, Y_tmp, deg=deg)
        beta_est2 = cross_moment(W_tmp, D_tmp, Y_tmp, deg=deg)
        
        betas_est1[it] = beta_est1
        betas_est2[it] = beta_est2
    
    beta_est1 = np.mean(betas_est1)
    beta_est2 = np.mean(betas_est2)
    std1 = np.std(betas_est1)
    if std1==0: 
        return beta_est1    
    std2 = np.std(betas_est2)
    if std2==0:
        print(beta_est2)
        return beta_est2
    coef1 = 1./std1**2
    coef2 = 1./std2**2
    beta = (coef1*beta_est1 + coef2*beta_est2) / (coef1 + coef2)
    return beta


# Remark: data - before withening
def init_w_guess_(data, g, latent, observed):
    up_data = data.t()
    w = torch.zeros(len(g.edges()))
    mask = torch.zeros(len(g.edges()))

    for i, e in enumerate(g.edges()):
        if e[0] < latent:
            w[i] = torch.Tensor(1).normal_().item()
            mask[i] = 1
        else:
            G_cov = up_data.cov()
            w[i] = G_cov[e[0]-latent,e[1]-latent]/G_cov[e[0]-latent,e[0]-latent]
            up_data[:, e[1]-latent] = up_data[:, e[1]-latent]-w[i]*up_data[:,e[0]-latent]

            an_s = sorted(nx.ancestors(g, e[0]))
            i_s = intersection(an_s,list(range(latent)))

            if len(i_s) > 0 :
                an_t = sorted(nx.ancestors(g, e[1]))
                i_t = intersection(an_t,list(range(latent)))
                if len(i_t)>0:
                    ints = intersection(i_s,i_t)
                    if len(ints)>0:
                        mask[i] = 1
    return w, mask


def tyc20(Z, W, D, Y):
    data_tmp = np.hstack((Z[:, None], D[:, None]))
    reg1 = LinearRegression().fit(data_tmp, W)
    W_new = reg1.predict(data_tmp)

    data_tmp = np.hstack((D[:, None], W_new[:, None]))
    reg2 = LinearRegression().fit(data_tmp, Y)
    return reg2.coef_[0]


def graphical_rica(latent, observed, g, data, data_whitening, epochs, lr, W_w, w_init, w_true, momentum=0, lmbda=0):
    
    """
        Graphical adaptation of RICA
        
        Parameters:
            - latent (int): Number of hidden variables.
            - observed (int): Number of observed variables.
            - g (nx.DiGraph): The DAG as a NetworkX DiGraph object.
            - data (torch.Tensor): Input data.
            - lr(double): Learning rate of the optimizer.
            - epochs (int): Number of optimization epochs.
            - W_w (torch.Tensor): Whitening matrix.
            - w_init (str): Weight initialization strategy ('random', 'true', 'cov_guess').
            - w_true (torch.Tensor): True weights of the DAG edges.
        
        Returns:
            - loss_data (torch.Tensor): Loss data during optimization.
            - w_loss_data (torch.Tensor): Squared distance of the difference between the true and the estimated parameters during optimization.
            
        """
    
    loss_data = torch.zeros(epochs)
    w_loss_data = torch.zeros(len(w_true), epochs)
                    
    mask = None
    if w_init=='cov_guess':
        w, mask = init_w_guess_(data, g, latent, observed)
        weight_true = w_true[mask==1]
        weight = Parameter(w[mask==1])
        fix_weight = w[mask == 0]        
        c_list = count_lists(mask)
    elif w_init=="true":
        weight_true = w_true
        weight = Parameter(torch.clone(w_true).detach().requires_grad_(True))
    else:
        weight_true = w_true
        weight = Parameter(torch.Tensor(len(g.edges())).normal_(0,1))

    optimizer = torch.optim.RMSprop([weight], lr, momentum=momentum)

    min_loss = None
    for epoch in range(epochs):
        adj = torch.eye(len(g.nodes()))
        if w_init == 'cov_guess':
            for ii, e in enumerate(g.edges()):
                if mask[ii] == 1:
                    adj[e]=-weight[int(c_list[ii])]
                else:
                    adj[e]=-fix_weight[int(c_list[ii])] 
        else:
            for e in range(len(g.edges())):
                adj[list(g.edges)[e]]=-weight[e]

        B = (torch.inverse(adj)).t()
        B = B[latent:latent+observed,:]
        B = W_w.matmul(B)

        latents = data_whitening.matmul(B)
        output = latents.matmul(B.t())
    
        diff = output - data_whitening
        loss_recon = 0
        if lmbda!=0:
            loss_recon = (diff * diff).mean()
        loss_latent = latents.abs().mean()
        loss = lmbda * loss_recon + loss_latent
        
        loss_data[epoch] = (loss.data).item()
        if min_loss is None or min_loss>loss_data[epoch]:
            min_loss = loss_data[epoch]
            weight_pred = weight.detach().clone()
        
        if  w_init == 'cov_guess':
            w_loss_data[mask==1, epoch] = (weight-w_true[mask==1].detach()).abs()
            w_loss_data[mask==0, epoch] = (fix_weight-w_true[mask==0].detach()).abs()
        else:
            w_loss_data[:, epoch] = (weight-w_true.detach().item).abs()       
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_data, w_loss_data, weight_pred, weight_true