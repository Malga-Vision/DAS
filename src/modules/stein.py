import torch
import time

from cdt.utils.R import launch_R_script
from modules.utils import *


def Stein_hess_diag(X, eta_G, eta_H, s = None):
    """
    Estimates the diagonal of the Hessian of log p_X at the provided samples points
    X, using first and second-order Stein identities
    """
    # n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    n = X_diff.shape[0]
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)
    K_inv = torch.inverse(K + eta_H * torch.eye(n))
    
    nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
    
    return -G**2 + torch.matmul(K_inv, nabla2K), G, K, K_inv, s


def Stein_hess_row(X, s, l, G, K, K_inv):
    """
    v-th row of the Hessian matrix
    """    
    X_diff = X.unsqueeze(1)-X
    Gl = torch.einsum('i,ij->ij', G[:,l], G)
    
    nabla2lK = torch.einsum('ik,ikj,ik->ij', X_diff[:,:,l], X_diff, K) / s**4
    nabla2lK[:,l] -= torch.einsum("ik->i", K) / s**2
    
    return -Gl + torch.matmul(K_inv, nabla2lK)


def fast_parents(hess_l, K, threshold, remaining_nodes):
        hess_m = hess_l.mean(dim=0)
        parents = []
        t = 0
        k = min(K, len(remaining_nodes))
        topk_values, topk_indices = torch.topk(hess_m, k, sorted=False)
        for j in range(k):
            if topk_values[j] > t:
                node = topk_indices[j]
                parents.append(remaining_nodes[node])
        return parents



def compute_top_order(X, eta_G, eta_H,  n_parents, threshold, normalize_var=True, dispersion="var"):
    n, d = X.shape
    A = np.zeros((d,d))
    order = []
    active_nodes = list(range(d))
    tot = 0
    for _ in range(d-1):
        start = time.time()
        H, G, K, K_inv, s = Stein_hess_diag(X, eta_G, eta_H)
        tot += time.time() - start
        if normalize_var:
            H = H / H.mean(axis=0)
        if dispersion == "var": # The one mentioned in the paper
            l = int(H.var(axis=0).argmin())
        elif dispersion == "median":
            med = H.median(axis = 0)[0]
            l = int((H - med).abs().mean(axis=0).argmin())
        else:
            raise Exception("Unknown dispersion criterion")
            
        # parents
        # H_l = Stein_hess_row(X, s, l, G, K, K_inv)
        H_l = Stein_hess_col(X, G, K, l, s, eta_G, n)
        parents = fast_parents(H_l, n_parents, threshold, active_nodes)
        A[parents, l] = 1
        A[l, l] = 0
        order.append(active_nodes[l])
        active_nodes.pop(l)

        X = torch.hstack([X[:,0:l], X[:,l+1:]])

    order.append(active_nodes[0])
    order.reverse()
    return order, A


def heuristic_kernel_width(X):
    """
    Estimator of width parameter for gaussian kernel
    """
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
    print(arguments)

    if prune_only:
        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return A
        dag = launch_R_script("../R_code/cam_pruning.R", arguments, output_function=retrieve_result)
        return dag
    else:
        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            Afull = pd.read_csv(arguments['{ADJFULL_RESULTS}']).values
            
            return A, Afull
        dag, dagFull = launch_R_script("/Users/user/Documents/EPFL/PHD/Causality/score_based/CAM.R", arguments, output_function=retrieve_result)
        top_order = fullAdj2Order(dagFull)
        return dag, top_order
        
  
def SCORE(X, eta_G=0.001, eta_H=0.001, normalize_var=False, dispersion="var", threshold=0.1, K=None):
    start_time = time.time()
    top_order, A_SCORE = compute_top_order(X, eta_G, eta_H, K, threshold, normalize_var, dispersion)
    SCORE_time = time.time() - start_time

    return A_SCORE, top_order, SCORE_time


def sortnregress(X, cutoff=0.001):
    var_order = np.argsort(X.var(axis=0))
    
    return cam_pruning(full_DAG(var_order), X, cutoff), var_order


def num_errors(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err
