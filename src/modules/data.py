import numpy as np
import torch

from utils import SHD

def dream_data(path_to_data):
    """
    TSV separated gene expression data
    Real data: Network 1-3
    Synthetic: Network 4
    """
    X = np.genfromtxt(fname=path_to_data, delimiter="\t", skip_header=1, filling_values=1)
    return torch.tensor(X)

def dream_adj(gold_standard_path, shape):
    """
    gold_standard_path (str): path to ground truth data
    """
    _, n = shape
    A = np.zeros((n, n))
    with open(gold_standard_path, 'r') as f:
        for line in f:
            values = line.split('\t')
            x = int(values[0][1:]) -1
            y = int(values[1][1:]) -1
            if int(values[2]) == 1:
                A[x, y] = 1
        
    return A