# -*- coding: utf-8 -*-

import os
import numpy as np
from cert_utils import varying_par_sim_full_graph, plot_cert_data, running_time_plot

# Set to False if you don't want to parallelize
parallel_exc = True

# List of different graph sizes to explore
p_sizes = [20, 25, 50, 75] + list(np.arange(100, 1001, 100))


# Generate datasets for varying graph sizes using varying_par_sim_full_graph function

datasets = [varying_par_sim_full_graph(p_total=p, n_reps=100, parallel_exc=parallel_exc, save_file = True) for p in p_sizes]




par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
files_dir = par_dir + "/Data/Data_certification/Full_graph/"

graph_sizes = [20, 100]
image_file = f'../Figures/Certification/Full_graph/perc_ident_p_{graph_sizes[0]}_p_{graph_sizes[1]}.png'

# Plot identifiability percentages against density for known and unknown graphs
plot_cert_data(files_dir = files_dir,
                          image_file = image_file,
                          graph_sizes = graph_sizes,
                          n_reps=100)

image_file = '../Figures/Certification/Full_graph/identifiability_running_time_random_edge.png'
running_time_plot(files_dir = files_dir,
                             image_file = image_file)
