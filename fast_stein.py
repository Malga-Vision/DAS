#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import torch
from utils import *
from stein import *

def adj(parents, d):
    A = np.zeros((d, d))
    # Optimize without double for loop
    for node in range(d):
        try:
            for p in parents[node]:
                A[node, p] = 1
        except KeyError:
            pass

    return A



def parent_from_var(var, order, d, threshold):
    """
    Assolutamente da rivedere, fatta a caso e il criterio di selezione non sarà questo
    """
    min_value = torch.min(var)
    augmented_var = torch.zeros(d)
    var_index = 0
    for node in range(d):
        if node in order:
            augmented_var[node] = min_value
        else:
            augmented_var[node] = var[var_index]
            var_index += 1

    parents = []
    parents_mask = augmented_var.ge(threshold)
    for i in range(parents_mask.shape[0]):
        if parents_mask[i]:
            parents.append(i)

    return parents


def hessian_difference(H_old, H_new, last_child):
    """
    Arguments:
        H_old: Diagonal of the Hessian at step t. H_old.size() = (n, r+1)
        H_new: Diagonal of the Hessian at step t+1. H_new.size() = (n, r)
        last_child: last node inserted in the topological order.

    Return: 
        A (n, d) matrix with the elementwise difference between elements of H_old and H_new.
    """

    # Remove last_child column H_old
    H_old = torch.hstack((H_old[:,:last_child], H_old[:, last_child+1: ]))

    # Should I consider other ways to compute the distance rather than a simple difference?

    # Elementwise difference and summary statistics
    dist = torch.abs(H_old - H_new)
    mean, std = torch.std_mean(dist, dim=0)
    var = torch.var(dist, dim=0)
    norm = torch.norm(dist, dim=0)

    return mean, std, var, norm


# Fa cacare riscrivere la stessa funzione quasi uguale, ma per ora preferisco lasciare il suo codice intonso
def compute_top_order_fast(X, eta_G, eta_H, normalize_var=True, dispersion="var"):
    """
    Estimate topological order from input data (Stein ridge regression)
    """
    n, d = X.shape
    order = []
    parents = dict()
    active_nodes = list(range(d))
    for i in range(d-1):
        H = Stein_hess(X, eta_G, eta_H) # Diagonal of the Hessian for each sample
        if normalize_var:
            H = H / H.mean(axis=0)
        if dispersion == "var": # The one mentioned in the paper
            l = int(H.var(axis=0).argmin())
        elif dispersion == "median":
            med = H.median(axis = 0)[0]
            l = int((H - med).abs().mean(axis=0).argmin())
        else:
            raise Exception("Unknown dispersion criterion")

        if i > 0:
            mean, std, var, norm = hessian_difference(H_old, H, l)

            # Try also with mean. 
            # Also, try with norm
            parents[order[-1]] = parent_from_var(var, order, d, 1)


        H_old = H
        order.append(active_nodes[l])
        active_nodes.pop(l)
        X = torch.hstack([X[:,0:l], X[:,l+1:]])
    order.append(active_nodes[0])
    order.reverse()
    
    return order, adj(parents, d)