from stein import *
import cdt

def generate(d, s0, N, noise_std = 1, noise_type = 'Gauss', graph_type = 'ER', GP = True, lengthscale=1):
    """
        Args:
            d (int): num of nodes
            s0 (int): expected num of edges
            graph_type (str): ER, SF
    """
    adjacency = simulate_dag(d, s0, graph_type, triu=True)
    teacher = Dist(d, noise_std, noise_type, adjacency, GP = GP, lengthscale=lengthscale)
    X, noise_var = teacher.sample(N)
    return X, adjacency


# Data generation paramters
graph_type = 'ER'
d = 200
s0 = 4*d
N = 1000

X, adj = generate(d, s0, N, GP=True)

# SCORE hyper-parameters
eta_G = 0.001
eta_H = 0.001
cam_cutoff = 0.001
pruning = "Fast"
threshold = 0.1 # None for CAM

# Test: provo a elevare a esponenziale la funzione discriminatoria,e  diminuire accordingly threshold
start = time.time()
A_SCORE, top_order_SCORE, _, _ =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning=pruning, threshold = threshold)
print(f"Excution time: ----- {round(time.time() - start, 2)}s -----")
print(f"Pruning: {pruning}, Threshold: {threshold}")
print("SHD : {}".format(SHD(A_SCORE, adj)))
print("SID: {}".format(int(cdt.metrics.SID(target=adj, pred=A_SCORE))))
print("top order errors: {}".format(num_errors(top_order_SCORE, adj)))
