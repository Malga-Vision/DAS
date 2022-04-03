from stein import *
from fast_stein import fast_SCORE, stein_vs_fast_pruning
import cdt
import time


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


# Data generation parameters
graph_type = 'ER'
d = 50
s0 = 10
N = 1000

X, adj = generate(d, s0, N, GP=True)

start_time = time.time()

# SCORE hyper-parameters
threshold = 0.2
eta_G = 0.001
eta_H = 0.001
cam_cutoff = 0.001

start = time.time()
A_SCORE, top_order_SCORE =  fast_SCORE(X, eta_G, eta_H, cam_cutoff, pruning="Stein", threshold = threshold)
print("SHD : {}".format(SHD(A_SCORE, adj)))
print("SID: {}".format(int(cdt.metrics.SID(target=adj, pred=A_SCORE))))
print("top order errors: {}".format(num_errors(top_order_SCORE, adj)))

print("--- %s seconds ---" % (time.time() - start))

def newline(f, n=1):
    for i in range(n):
        f.write("\n")

with open('results.txt', 'w') as f:
    # Intuition: Threshold must increase while increasing number of nodes
    # as more sum yield more noisy hessian values
    f.write(f"Threshold: {threshold}")
    A_SCORE = A_SCORE.astype(np.int) 
    A_diff = adj - A_SCORE

    if d <= 20:
        newline(f, 2)
        f.write(f"DIFFERENCE \n") # 1: threshold too high. -1: threshold too low
        for row in A_diff:
            f.write(f"{row}\n")

        newline(f)
        f.write(f"GROUND TRUTH \n")
        for row in adj:
            f.write(f"{row}\n")

        newline(f)
        f.write(f"A_SCORE \n")
        for row in A_SCORE:
            f.write(f"{row}\n")

    newline(f)
    tot_difference = np.sum(np.asmatrix(np.abs(A_diff)))
    num_links = np.sum(np.asmatrix(np.abs(adj)))
    f.write(f"Number of links: {num_links}")
    newline(f)
    f.write(f"Number of missed/wrong: {tot_difference}")
    


# A_stein, A_fast, top_order_SCORE = stein_vs_fast_pruning(X, eta_G, eta_H, cam_cutoff, threshold_f = 0.15, threshold_s = 0.1)

# # Stein
# print("SHD Stein pruning: {}".format(SHD(A_stein, adj)))
# print("SID Stein pruning: {}".format(int(cdt.metrics.SID(target=adj, pred=A_stein))))

# # Fast
# print("SHD Fast pruning: {}".format(SHD(A_fast, adj)))
# print("SID Fast pruning: {}".format(int(cdt.metrics.SID(target=adj, pred=A_fast))))

# print("top order errors: {}".format(num_errors(top_order_SCORE, adj)))