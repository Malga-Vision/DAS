#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from logging import PlaceHolder
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


def Stein_hess_col(X_diff, G, K, v, s, eta, n):
    """
        v-th col of the Hessian: vector of partial derivatives of score_v over all nodes

        Return: 
            Hess_v: 
    """
    Gv = torch.einsum('i,ij->ij', G[:,v], G)
    nabla2vK = torch.einsum('ik,ikj,ik->ij', X_diff[:,:,v], X_diff, K) / s**4
    nabla2vK[:,v] -= torch.einsum("ik->i", K) / s**2
    Hess_v = -Gv + torch.matmul(torch.inverse(K + eta * torch.eye(n)), nabla2vK)

    return Hess_v


# Should actually compute only the row of interest...
def Stein_hess_matrix(X, s, eta):
    """
    Compute the Stein Hessian estimator matrix for each sample in the dataset

    Args:
        X: N x D matrix of the data
        s: kernel width estimator
        eta: 

    Return:
        Hess: N x D x D hessian estimator of log(p(x))
    """
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('ikj,ik->ij', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta * torch.eye(n)), nablaK)
    
    # Compute the Hessian by column stacked together
    Hess = Stein_hess_col(X_diff, G, K, 0, s, eta, n) # Hessian of col 0
    Hess = Hess[:, None, :]
    for v in range(1, d):
        Hess = torch.hstack([Hess, Stein_hess_col(X_diff, G, K, v, s, eta, n)[:, None, :]])
    
    return Hess


# Actually, it is faster to do it while doing the topological computation.
# I recycle the diagonal elements. Next optimization step, is compute only the Hessian rows needed
def fast_pruning(X, top_order, eta_G, threshold):
    """
    Args:
        X: N x D matrix of the samples
        top_order: 1 x D vector of topoligical order. top_order[0] is source
        eta_g: regularizer coefficient
        threshold: Assign a parent for partial derivative greateq than threshold
    """
    d = X.shape[1]
    remaining_nodes = list(range(d))
    # Enforce acyclicity
    A = np.zeros((d,d))
    for i in range(d-1):
        l = top_order[-(i+1)]
        s = heuristic_kernel_width(X[:, remaining_nodes].detach())
        hess = Stein_hess_matrix(X[:, remaining_nodes], s, eta_G)
        hess_l = hess[:, remaining_nodes.index(l), :] # l-th row  N x D
        parents = [remaining_nodes[i] for i in torch.where(torch.abs(hess_l.mean(dim=0))> threshold)[0] if top_order[i] != l]

        A[parents, l] = 1
        A[l, l] = 0
        remaining_nodes.remove(l)
    return A


def compute_top_order_test(X, eta_G, eta_H, normalize_var=True, dispersion="var"):
    n, d = X.shape
    order = []
    active_nodes = list(range(d))
    for i in range(d-1):
        H = Stein_hess_diag(X, eta_G, eta_H)

        s = heuristic_kernel_width(X)
        placeholder = Stein_hess_matrix(X, s, eta_G)
        
        if normalize_var:
            H = H / H.mean(axis=0)

        if dispersion == "var": # The one mentioned in the paper
            l = int(H.var(axis=0).argmin())
        elif dispersion == "median":
            med = H.median(axis = 0)[0]
            l = int((H - med).abs().mean(axis=0).argmin())
        else:
            raise Exception("Unknown dispersion criterion")
        order.append(active_nodes[l])
        active_nodes.pop(l)
        X = torch.hstack([X[:,0:l], X[:,l+1:]])
    order.append(active_nodes[0])
    order.reverse() # First node(s): source 
    return order



def fast_SCORE(X, eta_G=0.001, eta_H=0.001, cutoff=0.001, normalize_var=False, dispersion="var", pruning = 'CAM', threshold=0.1):
    top_order = compute_top_order_test(X, eta_G, eta_H, normalize_var, dispersion)

    if pruning == 'CAM':
        return cam_pruning(full_DAG(top_order), X, cutoff), top_order
    elif pruning == 'Stein':
        return Stein_pruning(X, top_order, eta_G, threshold = threshold), top_order
    elif pruning == "Fast":
        return fast_pruning(X, top_order, eta_G, threshold=threshold), top_order
    else:
        raise Exception("Uexisting pruning method")




# # Fa cacare riscrivere la stessa funzione quasi uguale, ma per ora preferisco lasciare il suo codice intonso
# def compute_adj_fast(X, eta_G, eta_H, normalize_var=True, dispersion="var"):
#     """
#     Estimate topological order from input data (Stein ridge regression)
#     """
#     n, d = X.shape
#     order = []
#     active_nodes = list(range(d))
#     for i in range(d-1):
#         H = Stein_hess_diag(X, eta_G, eta_H) # Diagonal of the Hessian for each sample
#         if normalize_var:
#             H = H / H.mean(axis=0)
#         if dispersion == "var": # The one mentioned in the paper
#             l = int(H.var(axis=0).argmin())
#         elif dispersion == "median":
#             med = H.median(axis = 0)[0]
#             l = int((H - med).abs().mean(axis=0).argmin())
#         else:
#             raise Exception("Unknown dispersion criterion")

#         # Method 1 : compute the difference on the diagonal of the hessian, and use statistics to infer parents
#         if i > 0:
#             mean, std, var, norm = hessian_difference(H_old, H, l)
#             # parents[order[-1]] = parent_from_var(var, order, d, 1)
#             # TODO: Store statistics
#         H_old = H

#         order.append(active_nodes[l])
#         active_nodes.pop(l)
#         X = torch.hstack([X[:,0:l], X[:,l+1:]])
#     order.append(active_nodes[0])
#     order.reverse()
    
#     return order