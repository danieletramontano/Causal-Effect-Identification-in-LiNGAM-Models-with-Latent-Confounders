import torch
import numpy as np
import pandas as pd


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def count_lists(l):
    c_m = -1
    c_um = -1
    c_list = np.zeros(len(l))
    for i, e in enumerate(l):
        if l[i] == 0:
            c_um = c_um+1
            c_list[i] = c_um
        else:
            c_m = c_m+1
            c_list[i] = c_m
    return c_list


def save_torch_data(data, filename):
    data_numpy = data.numpy() #convert to Numpy array
    data_pandas = pd.DataFrame(data_numpy) #convert to a dataframe
    data_pandas.to_csv(filename, index=False, header=False) #save to file