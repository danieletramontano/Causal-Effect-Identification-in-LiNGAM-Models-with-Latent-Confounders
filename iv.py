#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:22:20 2023

@author: daniele
"""

import networkx as nx
import numpy as np


##ISTRUMENTAL VARIABLES

iv_adj = np.array([[0,0,1,1],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
g = nx.DiGraph(iv_adj)
nx.draw_networkx(g)

latent = 1
observed = 3
n = 10000
lambdas = [0,10]
moms = [0,1]
lr = 0.01
epochs = 100
w_init = 'given'
pseudoinverse = False
withening = True

data, params, W_with, w_true = hid_var_data(latent, observed, g ,n, withening)
            

loss_data, loss_recon_data, loss_latent_data, w_loss_data = graphical_rica(latent, observed, g, data, 
                                                                           moms, lambdas, epochs, lr, W_with, pseudoinverse,
                                                                           w_init, w_true) 

loss_plots(loss_data, loss_recon_data, loss_latent_data, w_loss_data,moms, lambdas, epochs)


