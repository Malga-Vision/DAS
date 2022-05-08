import GPy
import uuid
import os
import igraph as ig
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch.distributions import MultivariateNormal, Normal, Laplace, Gumbel

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

from cdt.metrics import retrieve_adjacency_matrix, SID

class Dist(object):
    def __init__(self, d, noise_std = 1, noise_type = 'Gauss', adjacency = None, GP = True, lengthscale = 1, f_magn = 1, GraNDAG_like = False):
        self.d = d
        if isinstance(noise_std, (int, float)):
            noise_std = noise_std * torch.ones(self.d)
        self.GP = GP
        self.lengthscale = lengthscale
        self.f_magn = f_magn
        self.GraNDAG_like = GraNDAG_like

        
        if self.GraNDAG_like:
            noise_std = torch.ones(d)
        
        if noise_type == 'Gauss':
            self.noise = Normal(0, noise_std) # give standard deviation
        elif noise_type == 'Laplace':
            self.noise = Laplace(0, noise_std / np.sqrt(2))
        elif noise_type == 'Gumbel':
            self.noise = Gumbel(0, np.sqrt(noise_std) * np.sqrt(6)/np.pi)
        else:
            raise NotImplementedError("Unknown noise type.")
        
        self.adjacency = adjacency
        if adjacency is None:
            self.adjacency = np.ones((d,d))
            self.adjacency[np.tril_indices(d)] = 0

        # Needs strictly upper triangular matrix

        assert(np.allclose(self.adjacency, np.triu(self.adjacency)))


    def sampleGP(self, X, lengthscale=1):
        ker = GPy.kern.RBF(input_dim=X.shape[1],lengthscale=lengthscale,variance=self.f_magn)
        C = ker.K(X,X)
        X_sample = np.random.multivariate_normal(np.zeros(len(X)),C)
        return X_sample


    def sample(self, n):
        noise = self.noise.sample((n,)) # n x d noise matrix
        X = torch.zeros(n, self.d)

        # !!! Only works if adjacency matrix is upper triangular !!!
        noise_var = np.zeros(self.d)
        if self.GP:
            for i in range(self.d):
                #print("Sample function number {}".format(i))
                parents = np.nonzero(self.adjacency[:,i])[0]
                if self.GraNDAG_like:
                    if len(parents) == 0: # For roots, noise variance sampled U(1,2)
                        noise_var[i] = np.random.uniform(1,2)
                    else: # Otherwise, noise variance sampled U(0.4,0.8)
                        noise_var[i] = np.random.uniform(0.4,0.8)
                    X[:, i] = np.sqrt(noise_var[i]) * noise[:,i]
                else:
                    X[:, i] = noise[:,i]
                if len(np.nonzero(self.adjacency[:,i])[0]) > 0:
                    X_par = X[:,parents]
                    X[:, i] += torch.tensor(self.sampleGP(np.array(X_par), self.lengthscale))
        else:
            for i in range(self.d):
                X[:, i] = noise[:,i]
                for j in np.nonzero(self.adjacency[:,i])[0]:
                    X[:, i] += torch.sin(X[:,j])
        return X, noise_var
    
    def log_p(self, X, active_nodes=None):
        if self.GP:
            raise NotImplementedError("Score computation not available with GPs.")
        if active_nodes is None:
            active_nodes = list(range(X.shape[1]))
        n = X.shape[0]
        d = X.shape[1]
        l = torch.zeros(n)
        for i, node_i in enumerate(active_nodes):
            fi = torch.zeros(n)
            for j, node_j in enumerate(active_nodes):
                if self.adjacency[node_j, node_i] != 0:
                    fi += torch.sin(X[:,j])
            l -= 0.5 * (X[:,i] - fi)**2
        return l



############## DAG ##############


def generate(d=None, s0=None, N=1000, noise_std = 1, noise_type = 'Gauss', graph_type = 'ER', GP = True, lengthscale=1):
    """
        Args:
            d (int): num of nodes
            s0 (int): expected num of edges
            graph_type (str): ER, SF
    """
    print("Generating data...", end=" ", flush=True)
    adjacency = simulate_dag(d, s0, graph_type, triu=True)
    teacher = Dist(d, noise_std, noise_type, adjacency, GP = GP, lengthscale=lengthscale)
    X, noise_var = teacher.sample(N)
    print("Done")
    return X, adjacency


def simulate_dag(d, s0, graph_type, triu=False):
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        if triu:
            return np.triu(B_und, k=1)
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False)
        B_und = _graph_to_adjmat(G)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    if not triu:
        B = _random_permutation(B)
    assert ig.Graph.Adjacency(B.tolist()).is_dag()
    return B


def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A


############## PRUNING ##############


def pns_(model_adj, x, num_neighbors, thresh):
    """Preliminary neighborhood selection"""
    num_samples = x.shape[0]
    num_nodes = x.shape[1]
    print("PNS: num samples = {}, num nodes = {}".format(num_samples, num_nodes))
    for node in range(num_nodes):
        print("PNS: node " + str(node))
        x_other = np.copy(x)
        x_other[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500)
        reg = reg.fit(x_other, x[:, node])
        selected_reg = SelectFromModel(reg, threshold="{}*mean".format(thresh), prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False).astype(np.float)

        model_adj[:, node] *= mask_selected

    return model_adj


############## METRICS ##############


def edge_errors(pred, target):
    """
    Counts all types of edge errors (false negatives, false positives, reversed edges)
    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)

    diff = true_labels - predictions

    rev = (((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn, fp, rev

def precision(e, fn, fp):
    tp = e - fn
    return tp / (tp+fp)

def recall(e, fn):
    tp = e - fn
    return tp / e


def SHD(pred, target):
    return sum(edge_errors(pred, target))


############## LOGGING ##############


def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output


def pretty_evaluate(pruning, threshold, adj, A_SCORE, top_order_err, SCORE_time, tot_time, sid, s0, K=None, pns=None):
    fn, fp, rev = edge_errors(A_SCORE, adj)
    d = A_SCORE.shape[0]
    precision_metric = precision(s0, fn, fp)
    recall_metric = recall(s0, fn)

    pretty = f"""
----------------------------------------------------

SCORE execution time:               {round(SCORE_time, 2)}s
Total execution time:               {round(tot_time, 2)}s

----------------------------------------------------

Pruning:                            {pruning}
K:                                  {K}
pns:                                {pns}
Threshold:                          {threshold}
Number of nodes:                    {d}

----------------------------------------------------

Topological order errors:           {top_order_err}
False negative:                     {int(fn)}
False positive:                     {int(fp)}
Recall:                             {round(recall_metric, 2)}
Precision:                          {round(precision_metric, 2)}
Reversed:                           {int(rev)}
SHD:                                {SHD(A_SCORE, adj)}
    """

    if sid:
        pretty += f"""
SID:                                {int(SID(target=adj, pred=A_SCORE))}
""".lstrip()

    return pretty


############## SERGIO-TESTING ##############
def ground_truth(d, path):
    A = np.zeros((d, d))
    ground_truth = pd.read_csv(path, header=None).to_numpy()
    for row in ground_truth:
        src, dest = row
        A[src, dest] = 1

    return A