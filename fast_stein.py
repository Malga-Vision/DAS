from logging import PlaceHolder
import torch
import time
from utils import *
from stein import *


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


# Would it be better to comptue only row of interest at each iteration?
def Stein_hess_matrix(X, s, eta):
    """
    Compute the Stein Hessian estimator matrix for each sample in the dataset

    Args:
        X: N x D matrix of the data
        s: kernel width estimator
        eta: regularization coefficient

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
    s = heuristic_kernel_width(X.detach()) # This actually changes at each iteration 
    hess = Stein_hess_matrix(X, s, eta_G)

    # TODO: Enforce acyclicity
    A = np.zeros((d,d))
    for i in range(d-1):
        l = top_order[-(i+1)]

        # Results are not actually better, while way slower
        # s = heuristic_kernel_width(X[:, remaining_nodes].detach())
        # hess = Stein_hess_matrix(X[:, remaining_nodes], s, eta_G)
        # hess_l = hess[:, remaining_nodes.index(l), :] # l-th row  N x D

        # N x remaining_nodees x remaining_nodes
        hess_remaining = hess[:, remaining_nodes, :]
        hess_remaining = hess_remaining[:, :, remaining_nodes]
        hess_l = hess_remaining[:, remaining_nodes.index(l), :]
        parents = []
        for j in torch.where(torch.abs(hess_l.mean(dim=0)) > threshold)[0]:
            if top_order[j] != l: # ?!
                parents.append(remaining_nodes[j])

        A[parents, l] = 1
        A[l, l] = 0
        remaining_nodes.remove(l)
    return A


def fast_SCORE(X, eta_G=0.001, eta_H=0.001, cutoff=0.001, normalize_var=False, dispersion="var", pruning = 'CAM', threshold=0.1):
    top_order = compute_top_order(X, eta_G, eta_H, normalize_var, dispersion)

    if pruning == 'CAM':
        return cam_pruning(full_DAG(top_order), X, cutoff), top_order
    elif pruning == 'Stein':
        return Stein_pruning(X, top_order, eta_G, threshold = threshold), top_order
    elif pruning == "Fast":
        return fast_pruning(X, top_order, eta_G, threshold=threshold), top_order
    else:
        raise Exception("Unexisting pruning method")


def stein_vs_fast_pruning(X, eta_G=0.001, eta_H=0.001, cutoff=0.001, normalize_var=False, dispersion="var", threshold_s=0.1, threshold_f=0.1):
    top_order = compute_top_order(X, eta_G, eta_H, normalize_var, dispersion)

    # Stein
    start_time = time.time()
    A_stein =  Stein_pruning(X, top_order, eta_G, threshold =threshold_s)
    print("Stein pruning: --- %s seconds ---" % (time.time() - start_time))

    # Fast
    start_time = time.time()
    A_fast  =  fast_pruning(X, top_order, eta_G, threshold=threshold_f)
    print("Fast pruning: --- %s seconds ---" % (time.time() - start_time))

    return A_stein, A_fast, top_order