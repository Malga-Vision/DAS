import time
import cdt
import pandas as pd

from sklearn.model_selection import ParameterGrid

from utils import *
from stein import *

class Experiment:
    def __init__(self):
        self.logs = pd.DataFrame()

    def run(self, d, threshold, N, graph_type, eta_G, eta_H, cam_cutoff, s0):        
        X, adj = generate(d, s0, N, GP=True)

        # SCORE hyper-parameters
        threshold = 0.2
        eta_G = 0.001
        eta_H = 0.001
        cam_cutoff = 0.001

        start_score = time.time()
        A_SCORE, top_order_SCORE =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning="Fast", threshold = threshold)
        score_time = time.time() - start_score

        print("SHD : {}".format(SHD(A_SCORE, adj)))
        print("SID: {}".format(int(cdt.metrics.SID(target=adj, pred=A_SCORE))))
        print("top order errors: {}".format(num_errors(top_order_SCORE, adj)))

        print("--- %s seconds ---" % (time.time() - start))
    


num_tests = 10 # multiple tests to compute mean and std scores 

# Output file
"fast_stein_log.txt"

# Tested params - s0 \in {d, 4*d}
# s0 \in {d, d*4}
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
d = [10, 20, 50, 100, 200]

# Fixed params
N = 1000
graph_type = 'EF'
eta_G = 0.001
eta_H = 0.001
cam_cutoff = 0.001


param_grid = {'d': d, 'threshold': thresholds}
params = list(ParameterGrid(param_grid))

experiments = Experiment()

for args in params:
    d = args['d']
    s0 = d
    threshold = args['threshold']
    for k in range(num_tests):
        experiments.run(d, threshold, N, graph_type, eta_G, eta_H, cam_cutoff)
