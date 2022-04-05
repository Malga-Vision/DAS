import torch
import time

from utils import *


def Stein_hess_diag(X, eta_G, eta_H, s = None):
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

# Try to compute Hess once and remove nodes each time
def compute_top_order(X, eta_G, eta_H, normalize_var=True, dispersion="var"):
    n, d = X.shape
    order = []
    active_nodes = list(range(d))
    for i in range(d-1):
        H = Stein_hess_diag(X, eta_G, eta_H)
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
    order.reverse()
    return order



def Stein_hess_parents(X, s, eta, l):
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('ikj,ik->ij', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta * torch.eye(n)), nablaK) # Expected: n x d, Ok
    Gl = torch.einsum('i,ij->ij', G[:,l], G)
    
    nabla2lK = torch.einsum('ik,ikj,ik->ij', X_diff[:,:,l], X_diff, K) / s**4
    nabla2lK[:,l] -= torch.einsum("ik->i", K) / s**2
    
    return -Gl + torch.matmul(torch.inverse(K + eta * torch.eye(n)), nabla2lK)


def heuristic_kernel_width(X):
    """
    Estimator of width parameter for gaussian kernel
    """
    X_diff = X.unsqueeze(1)-X
    D = torch.norm(X_diff, dim=2, p=2)
    s = D.flatten().median()
    return s


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
    print(arguments)

    if prune_only:
        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return A
        dag = launch_R_script("./cam_pruning.R", arguments, output_function=retrieve_result)
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
    start_time = time.time()
    top_order = compute_top_order(X, eta_G, eta_H, normalize_var, dispersion)
    SCORE_time = time.time() - start_time
    
    start_time = time.time()
    if pruning == 'CAM':
        A_SCORE = cam_pruning(full_DAG(top_order), X, cutoff)
    elif pruning == 'Stein':
        A_SCORE = Stein_pruning(X, top_order, eta_G, threshold = threshold)
    elif pruning == "Fast":
        A_SCORE = fast_pruning(X, top_order, eta_G, threshold=threshold)
    else:
        raise Exception("Unknown pruning method")

    return A_SCORE, top_order, SCORE_time, SCORE_time + (time.time() - start_time)


def sortnregress(X, cutoff=0.001):
    var_order = np.argsort(X.var(axis=0))
    
    return cam_pruning(full_DAG(var_order), X, cutoff), var_order


def num_errors(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err
