import numpy as np

from utils import SHD

def dream_data(path_to_data):
    """
    TSV separated gene expression data
    Real data: Network 1-3
    Synthetic: Network 4
    """
    X = np.genfromtxt(fname=path_to_data, delimiter="\t", skip_header=1, filling_values=1)
    return X

def dream_adj(gold_standard_path, shape):
    """
    gold_standard_path (str): path to ground truth data
    """
    _, n = X.shape
    A = np.zeros((n, n))
    with open(gold_standard_path, 'r') as f:
        for line in f:
            values = line.split('\t')
            x = int(values[0][1:]) -1
            y = int(values[1][1:]) -1
            if int(values[2]) == 1:
                A[x, y] = 1
        
    return A


path_to_data = "/data/francescom/dream5/DREAM5_network_inference_challenge/Network1/input_data/net1_expression_data.tsv"
gold_standard_path = "/data/francescom/dream5/DREAM5_network_inference_challenge/Evaluation_scripts/INPUT/gold_standard_edges_only/DREAM5_NetworkInference_Edges_Network1.tsv"

X = dream_data(path_to_data)
A = dream_adj(gold_standard_path, X.shape)

print("End")