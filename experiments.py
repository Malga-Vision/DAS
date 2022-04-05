import time
import cdt
import math
from numpy.core.fromnumeric import mean
import pandas as pd

from sklearn.model_selection import ParameterGrid

from utils import *
from stein import *

class Experiment:
    def __init__(self, num_tests, pruning):
        self.logs = []
        self.columns = [
            'V', 'E', 'N', 'threshold', 'fn', 'fp', 'reversed', 'SHD', 'SID' , 'D_top', 'SCORE time [s]','Total time [s]'
        ]
        self.num_tests = num_tests
        self.pruning=pruning
        self.output_file=f"./logs_{pruning.lower()}.csv"

    def run(self, d, s0, N, eta_G, eta_H, cam_cutoff=None, threshold=None):
        run_logs = []
        for k in range(self.num_tests):        
            X, adj = generate(d, s0, N, GP=True)

            A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, cam_cutoff, pruning=self.pruning, threshold = threshold)

            fn, fp, rev = edge_errors(A_SCORE, adj)
            SHD = sum((fn, fp, rev))
            SID = int(cdt.metrics.SID(target=adj, pred=A_SCORE))
            top_order_errors = num_errors(top_order_SCORE, adj)

            run_logs.append([d, s0, N, threshold, fn, fp, rev, SHD, SID, top_order_errors, SCORE_time, tot_time])

        mean_logs = np.mean(run_logs, axis=0)
        std_logs = np.std(run_logs, axis=0)
        logs = []
        for i in range(len(self.columns)):
            m = mean_logs[i]
            s = std_logs[i]
            if self.columns[i] in ["V", "E", "N"]:
                logs.append(f"{int(m)}")
            elif self.columns[i] == "threshold":
                logs.append(f"{round(m, 2)}")
            else:
                logs.append(f"{round(m, 2)} +- {round(s, 2)}")
        
        self.logs.append(logs)

    def save_logs(self):
        df = pd.DataFrame(self.logs, columns =self.columns)
        if self.pruning == "CAM":
            df.drop('threshold', axis=1)
        df.to_csv(self.output_file)

    def get_logs(self):
        """
        Convert logs list into dataframe
        """
        df = pd.DataFrame(self.logs, columns =self.columns)
        df.to_csv(self.output_file)
        return df


def stein_fast_exp(num_tests, N, eta_G, eta_H):
    # Hyperparameters
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
    d_values= [10, 20, 50, 100]
    param_grid = {'d': d_values, 'threshold': thresholds}
    params = list(ParameterGrid(param_grid))

    run = 0
    tot_runs = len(params)*2
    start = time.time()

    for pruning in ["Fast"]:
        experiments = Experiment(num_tests, pruning)
        for args in params:
            d = args['d']
            s0 = d # TODO Fix! Test con s0 fixed at 10, and s0 at 4*d
            threshold = args['threshold']
            experiments.run(d, s0, N, eta_G, eta_H, threshold=threshold)
            experiments.save_logs()

            run += 1
            print(f"Run {run}/{tot_runs} ------ {round(time.time() - start, 2)}s ------")


def cam_exp(num_tests, N, eta_G, eta_H, cam_cutoff):
    d_values = [10, 20, 50]

    run = 0
    tot_runs = len(d_values)
    start = time.time()
    experiments = Experiment(num_tests, "CAM")
    for d in d_values:
        s0 = d
        experiments.run(d, s0, N, eta_G, eta_H, cam_cutoff, 0)
        experiments.save_logs()

        run += 1
        print(f"Run {run}/{tot_runs} ------ {round(time.time() - start, 2)}s ------")
        

# Fixed params
N = 1000
graph_type = 'EF'
eta_G = 0.001
eta_H = 0.001
cam_cutoff = 0.001

num_tests = 2 # multiple tests to compute mean and std scores 
pruning_algorithm = "Fast"

if pruning_algorithm in ["Fast Stein", "Fast", "Stein"]:
    stein_fast_exp(num_tests, N, eta_G, eta_H)

elif pruning_algorithm == "CAM":
    cam_exp(num_tests, N, eta_G, eta_H, cam_cutoff)