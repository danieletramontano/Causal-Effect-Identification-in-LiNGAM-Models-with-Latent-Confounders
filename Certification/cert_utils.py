# -*- coding: utf-8 -*-
"""
This script contains functions to generate random Directed Acyclic Graphs (DAGs)
with hidden variables, and evaluate the identifiability of total causal effects
in the generated graphs.

Requirements:
- Python 3.10.9 
- NetworkX
- NumPy
- Multiprocess
- Pandas
- Matplotlib
"""

# Import necessary libraries

import time
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import multiprocess as mp
import pandas as pd
import sys

sys.path.append("../")
from Helpers.utils import minim_data, save_object, load_object

def hid_var_dag(latent=2,
                observed=5,
                prob_acc=0.5):
    """
        - Generates a random DAG with hidden variables.
        - Parameters:
             - latent (int): Number of hidden variables.
             - observed (int): Number of observed variables.
             - prob_acc (float): Probability of accepting an edge.
        - Returns:
             - g (nx.DiGraph): Generated DAG as a NetworkX DiGraph object.
    """

    adj = np.zeros((latent+observed, latent+observed))
    for i in range(latent+observed):
        for j in range(max(i+1, latent), latent+observed):
            if np.random.uniform(0, 1) < prob_acc:
                adj[i, j] = 1

    return nx.DiGraph(adj)



def check_pair_tce_id(graph=None,
                      edg=None,
                      latent=0):
    """
        - Checks the identifiability of one total causal effect in a graph.
        - Parameters:
             - graph (networkx.DiGraph): Graph.
             - edg (tuple): Edge in the graph to evaluate.
             - latent (int): Number of latent variables.
        - Returns:
            - Tuple of booleans (id_known, id_unknown) indicating identifiability status.
    """
    start_time = time.time()
    
    top_ord = list(nx.topological_sort(graph))
    
    
    edg = [top_ord.index(x) for x in edg]
    graph = nx.relabel_nodes(graph, lambda x: top_ord.index(x))
    id_known = True
    id_unknown = True
    k = 0
    
    
    de_j = sorted(nx.descendants(graph, edg[0]) | {edg[0]})
    
    if edg[1] not in de_j:
        end_time = time.time()
        tot_time = end_time - start_time
        return id_known, id_unknown, tot_time

    while id_known is True and k < latent:

        ch_k = sorted(graph.successors(k))
        if len(ch_k) > 1 and ch_k[0] == edg[0]:
            g_copy = graph.copy()
            pa_j = sorted(graph.predecessors(edg[0]))
            for par in pa_j:
                g_copy.remove_edge(par, edg[0])

            de_k_not_j = sorted(nx.descendants(g_copy, k))
            if edg[1] in de_k_not_j:
                de_k = sorted(nx.descendants(graph, k))
                if de_k == de_j:
                    id_known = False
                    id_unknown = False
                    ch_k_not_j = sorted(g_copy.successors(k))
                    for i in pa_j + [edg[0]]:
                        ch_i = sorted(g_copy.successors(i))
                        if sum(f in ch_i for f in ch_k_not_j) < len(ch_k_not_j):
                            id_known = True
        k += 1
        end_time = time.time()
        tot_time = end_time - start_time

    return id_known, id_unknown, tot_time




def full_graph_id(graph=None,
                  latent=0):
    """
        - Checks the identifiability of all the parameters in the graph.
        - Parameters:
             - graph (networkx.DiGraph): Graph to analyze.
             - latent (int): Number of latent variables.
        - Returns:
             - Tuple of booleans (id_known, id_unknown) indicating global identifiability status.
    """

    start_time = time.time()
    id_known = True
    id_unknown = True
    k = 0
    while id_known is True and k < latent:
        ch_k = sorted(graph.successors(k))

        if len(ch_k) > 1:
            j_k = ch_k[0]
            de_k = sorted(nx.descendants(graph, k))
            de_j = sorted(nx.descendants(graph, j_k) | {j_k})

            if de_j == de_k:
                id_unknown = False
                id_known = False

                g_copy = graph.copy()
                pa_j = sorted(graph.predecessors(j_k))

                for par in pa_j:
                    g_copy.remove_edge(par, j_k)

                ch_k_not_j = sorted(g_copy.successors(k))

                for i in pa_j + [j_k]:
                    ch_i = sorted(g_copy.successors(i))

                    if sum(f in ch_i for f in ch_k_not_j) < len(ch_k_not_j):
                        id_known = True
        k += 1

    end_time = time.time()
    tot_time = end_time-start_time

    return id_known, id_unknown, tot_time

def random_edge_id(graph=None,
                   latent=0):
    """
       - Checks the identifiability of the total causal effect associated to a random edge in the graph.
       - Parameters:
            - graph (networkx.DiGraph): Graph to analyze.
            - latent (int): Number of latent variables.
       - Returns:
            - Tuple of booleans (id_known, id_unknown) indicating the identifiability status.
            - tot_time (float): Time in seconds for the execution.
    """

    ran_edge = random.choice(list(graph.edges()))
    id_known, id_unknown, tot_time = check_pair_tce_id(graph=graph,
                                                       edg=ran_edge,
                                                       latent=latent)

    return id_known, id_unknown, tot_time


def varying_par_sim_full_graph(n_reps=10000,
                               n_probs=5,
                               p_total=25,
                               save_file=False,
                               parallel_exc=True):
    """
        - Simulates identification results for various parameters.
        - Parameters:
                - n_reps (int): Total number of repetitions for simulation.
                - n_probs (int): Number of different edge probability levels to explore.
                - p_total (int): Total number of variables in the generated graphs.
                - save_file (bool): If True, saves the simulation results to a file.
                - parallel_exc (bool): If True, runs simulations in parallel using multiprocessing.
        - Returns:
            - results (dataframe): Dataframe containing identification results for different parameters.

    """

    latent_observed_ratio = np.arange(0.1, 1, 0.2)

    id_df_list = []

    for lat_obs_ratio in latent_observed_ratio:
        p_latent = int(p_total*lat_obs_ratio)
        p_observed = int(p_total - p_latent)

        def fix_size_sim(i,
                         p_latent=p_latent,
                         p_observed=p_observed
                         ):

            id_unknown_prop = 0
            id_known_prop = 0
            time_ex = 0

            prob = 1/n_probs*(i+1)

            for _ in range(n_reps):
                g_latent = hid_var_dag(
                    latent=p_latent, observed=p_observed, prob_acc=prob)

                id_known, id_unknown, time_exc = full_graph_id(
                    g_latent, p_latent)

                id_unknown_prop += id_unknown
                id_known_prop += id_known
                time_ex += time_exc

            temp_results = {
                'Size_observed': p_observed,
                'Size_latent': p_latent,
                'Density': prob,

                'Time': time_ex,
                'Identifibility_known': id_known_prop,
                'Identifibility_unknown': id_unknown_prop,
            }

            pd_results = pd.DataFrame(temp_results, index=[0])

            return pd_results

        if parallel_exc is True:
            def parallel_sim_fix_size():
                if __name__ == '__main__':
                    pool = mp.get_context('spawn').Pool()
                    i_list = range(4)

                    results = pool.map(fix_size_sim, i_list)

                    # Step 3: Don't forget to close
                    pool.close()

                    return pd.concat(results)
                return None

            id_df_list.append(parallel_sim_fix_size())
        else:
            def sim_fix_size():

                results = [fix_size_sim(i) for i in range(4)]

                # Step 3: Don't forget to close

                return pd.concat(results)

            id_df_list.append(sim_fix_size())

    results = pd.concat(id_df_list, ignore_index=True)
    if save_file is True:
        save_object(minim_data(data=results,
                               head=f'../Data/Data_certification/Full_graph/p_{p_total}_n_{n_reps}'))
    return results


def running_time_plot(files_dir,
                    title_fontsize=24,
                    label_fontsize=24,
                    legend_fontsize=24,                                  
                    image_file = None,
                    n_reps = 100):
    
    """
    - Plots running time for full graph simulations against different parameters.
    - Parameters:
        - files_dir(str): Location of the files to process.
        - title_fontsize (int): Font size for the plot title.
        - label_fontsize (int): Font size for axis labels.
        - legend_fontsize (int): Font size for the legend.
        - image_file (str): If not None, name of image file.
        - n_reps (int): number of sampled graph for each setup.
    Returns:
        - Displays a plot visualizing the run time.
    """

    file_names = [f for f in os.listdir(files_dir) if os.path.isfile(files_dir+f) if os.path.splitext(f)[1] == '.pickle']
    files = []
    for file_name in file_names:
        files.append(load_object(files_dir + file_name).df)
           
    ax = plt.subplots(figsize=(10, 6))[1]
    ax.grid()
    ax.set_xlabel('Size', fontsize=label_fontsize)
    ax.set_ylabel('Time (s)', fontsize=label_fontsize)
    ax.set_title('Running time', fontsize=title_fontsize)
    data = pd.concat(files, ignore_index=True)
    data["Size"] = data["Size_observed"] + data["Size_latent"]
    data["Obs_ratio"] = data["Size_observed"]/data["Size"]
    data["Density"] = round(data["Density"], 2)
    lines = []

    for density in [0.2, 0.4, 0.6, 0.8]:
        data_filt = data.loc[(data["Density"] == density) &
                             (data["Obs_ratio"] <= 0.52) &
                             (data["Obs_ratio"] >= 0.5)]

        data_time_tot = np.flip(data_filt["Time"].to_numpy())
        data_time_mean = data_time_tot/n_reps

        data_size = np.flip(data_filt["Size"].to_numpy())
        order_vec = np.sort(np.array([data_size, data_time_mean]), axis=1)

        line, = ax.plot(order_vec[0, :],
                        order_vec[1, :],
                        linestyle='-',
                        marker='',
                        alpha=density,
                        color="blue",
                        label=f'{density}')
        lines.append(line)
        plt.xticks(fontsize = label_fontsize)
        plt.yticks(fontsize = label_fontsize)

    ax.legend(loc='upper left',
              title_fontsize = title_fontsize,
              fontsize=legend_fontsize,
              title="Prob. of Acc.")
    
    if image_file is not None:
        plt.savefig(image_file,
                    bbox_inches='tight')

    plt.show()


def plot_cert_data(files_dir,
                              graph_sizes = None,
                              n_reps=100,
                              title_fontsize=24,
                              label_fontsize=24,
                              legend_fontsize=24,
                              image_file=None):
    """
        Plots identifiability percentages against density for known and unknown graphs.

        - Parameters:
            - files_dir(str): Location of the files to process.
            - datasets (list): List of dataframes containing identification results for different parameters.
            - n_reps (int): Total number of repetitions for simulation.
            - title_fontsize (int): Font size for the plot title.
            - label_fontsize (int): Font size for axis labels.
            - legend_fontsize (int): Font size for the legend.
            - image_file (str): If not None, name of image file.
            - n_reps (int): number of sampled graph for each setup.

        Returns:
        - Displays a plot visualizing the identifiability percentages.
    """
    
    
    if graph_sizes is None:
        graph_sizes = [20, 100]


    file_names = [f for f in os.listdir(files_dir) if os.path.isfile(files_dir+f) if os.path.splitext(f)[1] == '.pickle']
    datasets = []
    for file_name in file_names:
        for graph_size in graph_sizes:
            if f'p_{graph_size}_' in file_name:
                datasets.append(load_object(files_dir + file_name).df)
        
    def custom_sort_key(data):
        return(np.unique(data['Size_observed'].to_numpy())[0]) 

    datasets.sort(key = custom_sort_key)

    fig, axes = plt.subplots(nrows=len(datasets), 
                             sharex = False, 
                             figsize=(10, 6))
    
    plt.subplots_adjust(hspace=0.45)

    
    fig.text(0.02, 0.5,
             '% Identifiable Parameters',
             va='center',
             rotation='vertical',
             fontsize=label_fontsize)

    p_totals = []

    for data_i, data in enumerate(datasets):
        
        ax = axes[data_i]
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.tick_params(axis = "x", labelsize=legend_fontsize)
        ax.tick_params(axis = "y", labelsize=legend_fontsize-2)
           
        ax.set_ylim(-0.05, 1.05)
        p_observed = data['Size_observed'][0]
        p_total = p_observed + data['Size_latent'][0]
        densities = data.loc[(data['Size_observed'] ==
                              p_observed)]['Density'].to_numpy()
        p_obs = np.unique(data['Size_observed'].to_numpy())
        p_totals.append(p_total)
        if data_i == len(datasets)-1:
            ax.set_xlabel('Probability of Acceptance', fontsize=label_fontsize)
        
        ax.set_title(f'p = {p_total}', fontsize=title_fontsize)

        lines = []
        colors = {'known': 'blue', 'unknown': 'green'}

        for p_observed in p_obs:
            data_filt = data.loc[(data['Size_observed'] == p_observed)]

            known_res = data_filt['Identifibility_known'].to_numpy()

            line1, = ax.plot(densities, known_res/n_reps,
                             linestyle='-',
                             marker="o",
                             color=colors["known"],
                             alpha=p_observed/p_total,
                             label=f'Known Graph, p_o/p = {round(p_observed/p_total,1)}')
            lines.append(line1)

            unknown_res = data_filt['Identifibility_unknown'].to_numpy()

            line1, = ax.plot(densities, unknown_res/n_reps,
                             linestyle='-',
                             marker="o",
                             color=colors["unknown"],
                             alpha=p_observed/p_total,
                             label=f'Unknown Graph, $p_o/p$ = {round(p_observed/p_total,1)}')
            lines.append(line1)
                

            
    hand_1 = [mpatches.Patch(color='blue',
                             label=f'{round(p_observed/p_total,1)}',
                             alpha=round(p_observed/p_total, 1)) for p_observed in p_obs]

    blue_patch = mpatches.Patch(color='blue', label='Known')
    green_patch = mpatches.Patch(color='green', label='Unknown')

    hand_2 = [blue_patch, green_patch]


    leg_1 = ax.legend(handles=hand_1,
              loc='center right',
              title="$p_o/p$",
              title_fontsize = title_fontsize-2,
              fontsize=legend_fontsize,
              bbox_to_anchor=(1.26, 0.67),
              ncols=1)

    ax.legend(handles=hand_2,
                      loc='center right',
                      title="Graph",
                      title_fontsize = title_fontsize-2,
                      fontsize=legend_fontsize,
                      bbox_to_anchor=(1.39, 2.03))


    ax.add_artist(leg_1)
    if image_file is not None:
        plt.savefig(image_file,
                    bbox_inches='tight')

    plt.show()


def varying_par_sim_random_edge(n_reps=10000,
                                n_probs=5,
                                p_total=25,
                                save_file=False,
                                parallel_exc=True):
    """
        - Simulates identification results for various parameters with random edges.
        - Parameters:
             - n_reps (int): Total number of repetitions for simulation.
             - n_probs (int): Number of different edge probability levels to explore.
             - p_total (int): Total number of variables in the generated graphs.
             - save_file (bool): If True, saves the simulation results to a file.
             - parallel_exc (bool): If True, runs simulations in parallel using multiprocessing.
        - Returns:
            - results (dataframe): Dataframe containing identification results for different parameters.

    """

    latent_observed_ratio = np.arange(0.1, 1, 0.2)

    id_df_list = []

    for lat_obs_ratio in latent_observed_ratio:
        p_latent = int(p_total*lat_obs_ratio)
        p_observed = int(p_total - p_latent)

        def fix_size_sim_par(i,
                             p_latent=p_latent,
                             p_observed=p_observed
                             ):

            id_unknown_prop = 0
            id_known_prop = 0
            time_ex = 0
            
            prob = 1/n_probs*(i+1)

            for _ in range(n_reps):
                len_g = 0
                while len_g == 0:
                    latent_graph = hid_var_dag(latent=p_latent,
                                               observed=p_observed,
                                               prob_acc=prob)
                    len_g = len(latent_graph.edges())

                    if list(latent_graph.edges())[-1][0] < p_latent:
                        len_g = 0

                source_node = 0

                while source_node < p_latent:
                    ran_edge = random.choice(list(latent_graph.edges()))
                    source_node = ran_edge[0]

                id_known, id_unknown, time_exc = check_pair_tce_id(graph=latent_graph,
                                                                   edg=ran_edge,
                                                                   latent=p_latent)
                

            

                id_unknown_prop += id_unknown
                id_known_prop += id_known
                time_ex += time_exc
                

            temp_results = {
                'Size_observed': p_observed,
                'Size_latent': p_latent,
                'Density': prob,

                'Time': time_ex,
                'Identifibility_known': id_known_prop,
                'Identifibility_unknown': id_unknown_prop,
                
                
            }

            pd_results = pd.DataFrame(temp_results, index=[0])

            return pd_results
        if parallel_exc is True:
            def parallel_sim_fix_size():
                if __name__ == '__main__':
                    pool = mp.get_context('spawn').Pool()
                    i_list = range(4)

                    results = pool.map(fix_size_sim_par, i_list)

                    pool.close()

                    return pd.concat(results)

                return None

            id_df_list.append(parallel_sim_fix_size())
        else:
            def sim_fix_size():
                results = [fix_size_sim_par(i) for i in range(4)]

                return pd.concat(results)

            id_df_list.append(sim_fix_size())

    results = pd.concat(id_df_list, ignore_index=True)
    if save_file is True:
        save_object(minim_data(data=results,
                               head=f'../Data/Data_certification/Random_edge/p_{p_total}_n_{n_reps}'))
    return results
