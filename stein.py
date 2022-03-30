#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import torch
from utils import *

    
def Stein_hess(X, eta_G, eta_H, s = None):
    """
    Estimates the diagonal of the Hessian of log p_X at the provided samples points
    X, using first and second-order Stein identities
    """
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)
    
    nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
    return -G**2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n)), nabla2K)


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


def compute_top_order(X, eta_G, eta_H, normalize_var=True, dispersion="var"):
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
    
def Stein_pruning(X, top_order, eta, threshold = 0.1):
    d = X.shape[1]
    remaining_nodes = list(range(d))
    A = np.zeros((d,d))
    for i in range(d-1):
        l = top_order[-(i+1)]
        s = heuristic_kernel_width(X[:, remaining_nodes].detach())
        p = Stein_hess_parents(X[:, remaining_nodes].detach(), s, eta, remaining_nodes.index(l))
        p_mean = p.mean(axis=0).abs()
        s_l = 1 / p_mean[remaining_nodes.index(l)]
        parents = [remaining_nodes[i] for i in torch.where(p_mean > threshold / s_l)[0] if top_order[i] != l]
        #parents = torch.where(p.mean(axis=0) > 0.1)[0]
        A[parents, l] = 1
        A[l, l] = 0
        remaining_nodes.remove(l)
    return A

def Stein_hess_parents(X, s, eta, l):
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('ikj,ik->ij', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta * torch.eye(n)), nablaK)
    Gl = torch.einsum('i,ij->ij', G[:,l], G)
    
    nabla2lK = torch.einsum('ik,ikj,ik->ij', X_diff[:,:,l], X_diff, K) / s**4
    nabla2lK[:,l] -= torch.einsum("ik->i", K) / s**2
    
    return -Gl + torch.matmul(torch.inverse(K + eta * torch.eye(n)), nabla2lK)

def heuristic_kernel_width(X):
    X_diff = X.unsqueeze(1)-X
    D = torch.norm(X_diff, dim=2, p=2)
    s = D.flatten().median()
    return s

def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order

def cam_pruning(A, X, cutoff, prune_only=True, pns=False):
    save_path = "./"

    data_np = np.array(X.detach().cpu().numpy())
    data_csv_path = np_to_csv(data_np, save_path)
    dag_csv_path = np_to_csv(A, save_path)

    arguments = dict()
    arguments['{PATH_DATA}'] = data_csv_path
    arguments['{PATH_DAG}'] = dag_csv_path
    arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
    arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
    arguments['{CUTOFF}'] = str(cutoff)
    arguments['{VERBOSE}'] = "TRUE"

    if prune_only:
        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return A

        dag = launch_R_script(f"{os.getcwd()}/cam_pruning.R", arguments, output_function=retrieve_result)
        return dag
    else:
        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            Afull = pd.read_csv(arguments['{ADJFULL_RESULTS}']).values
            
            return A, Afull
        dag, dagFull = launch_R_script("/Users/user/Documents/EPFL/PHD/Causality/score_based/CAM.R", arguments, output_function=retrieve_result)
        top_order = fullAdj2Order(dagFull)
        return dag, top_order
        
  

  
def SCORE(X, eta_G=0.001, eta_H=0.001, cutoff=0.001, normalize_var=False, dispersion="var", pruning = 'CAM', threshold=0.1):
    top_order, my_adj = compute_top_order(X, eta_G, eta_H, normalize_var, dispersion)
    return my_adj

    if pruning == 'CAM':
        return cam_pruning(full_DAG(top_order), X, cutoff), top_order, my_adj
    elif pruning == 'Stein':
        return Stein_pruning(X, top_order, eta_G, threshold = threshold), top_order, my_adj
    else:
        raise Exception("Unknown pruning method")

def sortnregress(X, cutoff=0.001):
    var_order = np.argsort(X.var(axis=0))
    
    return cam_pruning(full_DAG(var_order), X, cutoff), var_order


def num_errors(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err


def Stein_grad(X, s, eta): # Not used
    n, d = X.shape

    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s

    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    return torch.matmul(torch.inverse(K + eta * torch.eye(n)), nablaK)
