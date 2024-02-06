#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np


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


class minim_data:
    def __init__(self, data, head):
        self.df = data
        self.head = head

def save_object(obj):
    try:
        with open(obj.head+".pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
        
def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
