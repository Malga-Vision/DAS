import abc
import time
import cdt
import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid

from utils import *
from stein import *
from datasets import *

class Experiment(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run_config(self, params, N, eta_G, eta_H):
        """Run self.num_tests experiment on a given configuration"""
        raise NotImplementedError

    @abc.abstractmethod
    def run_experiment(num_tests, N, eta_G, eta_H):
        """Run experiments on all configurations for a given pruning algorithm"""
        raise NotImplementedError

    @abc.abstractmethod
    def config_logs(self, run_logs):
        """Logs of a tested configuration"""
        raise NotImplementedError

    def metrics(self, A_SCORE, adj, top_order_SCORE, compute_SID=True):
            fn, fp, rev = edge_errors(A_SCORE, adj)
            SHD = sum((fn, fp, rev))

            SID = -1
            if compute_SID:
                SID = int(cdt.metrics.SID(target=adj, pred=A_SCORE))
            top_order_errors = num_errors(top_order_SCORE, adj)
            return fn, fp, rev, SHD, SID, top_order_errors

    def save_logs(self):
        df = pd.DataFrame(self.logs, columns =self.columns)
        df.to_csv(self.output_file)



class CAMExperiment(Experiment):
    def __init__(self, d_values, num_tests, s0, data_type, cam_cutoff, output_file):
        self.d_values = d_values
        self.num_tests = num_tests
        self.s0 = s0
        self.data_type = data_type
        self.cam_cutoff = cam_cutoff

        self.logs = []
        self.columns = [
            'V', 'E', 'N', 'fn', 'fp', 'reversed', 'SHD', 'SID' , 'D_top', 'SCORE time [s]','Total time [s]'
        ]
        self.output_file = output_file

    def config_logs(self, run_logs):
        mean_logs = np.mean(run_logs, axis=0)
        std_logs = np.std(run_logs, axis=0)
        logs = []
        for i in range(len(self.columns)):
            m = mean_logs[i]
            s = std_logs[i]
            if self.columns[i] in ["V", "E", "N"]:
                logs.append(f"{int(m)}")
            else:
                logs.append(f"{round(m, 2)} +- {round(s, 2)}")
        self.logs.append(logs)
        

    def run_config(self, params, N, eta_G, eta_H):
        d = params['d']
        if self.s0 == 'd':
            s0 = d
        elif self.s0 == '4d':
            s0 = 4*d
        else:
            raise ValueError("Forbidden s0 value")
        
        run_logs = []
        for k in range(self.num_tests):
            X, adj = generate(d, s0, N, noise_type=self.data_type, GP=True)

            A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, self.cam_cutoff, pruning="CAM")
            fn, fp, rev, SHD, SID, top_order_errors = self.metrics(A_SCORE, adj, top_order_SCORE)
            run_logs.append([d, s0, N, fn, fp, rev, SHD, SID, top_order_errors, SCORE_time, tot_time])

        self.config_logs(run_logs)

    
    def run_experiment(self, N, eta_G, eta_H):
        start = time.time()
        run = 0
        tot_runs = len(self.d_values)
        for d in list(ParameterGrid({'d': self.d_values})):
            print(f"Running experiment for d: {d}")
            self.run_config(d, N, eta_G, eta_H)
            self.save_logs()

            run += 1
            print(f"Run {run}/{tot_runs} ------ {round(time.time() - start, 2)}s ------")


    def run_inference(self, path):
        X = dream_data(path)
        SCORE(X, eta_G, eta_H, pruning=self.pruning, threshold=threshold)
        return SCORE(X, eta_G, eta_H, pruning=self.pruning, threshold=threshold)


class FastExperiment(Experiment):
    def __init__(self, d_values, num_tests, s0, data_type, thresholds, pruning, output_file):
        self.d_values = d_values
        self.num_tests = num_tests
        self.s0 = s0
        self.data_type = data_type
        self.thresholds = thresholds
        self.pruning = pruning

        self.logs = []
        self.columns = [
            'V', 'E', 'N', 'threshold', 'fn', 'fp', 'reversed', 'SHD', 'SID' , 'D_top', 'SCORE time [s]','Total time [s]'
        ]
        self.output_file = output_file

    def config_logs(self, run_logs, compute_SID):
        mean_logs = np.mean(run_logs, axis=0)
        std_logs = np.std(run_logs, axis=0)
        logs = []
        for i in range(len(self.columns)):
            m = mean_logs[i]
            s = std_logs[i]
            if self.columns[i] in ["V", "E", "N"]:
                logs.append(f"{int(m)}")
            elif self.columns[i] == 'threshold':
                logs.append(round(m, 2))
            elif not compute_SID and self.columns[i] == "SID":
                logs.append(None)
            else:
                logs.append(f"{round(m, 2)} +- {round(s, 2)}")
        self.logs.append(logs)


    def run_config(self, params, N, eta_G, eta_H):
        d = params['d']
        threshold = params['threshold']

        compute_SID = True
        if d > 200:
            compute_SID = False
        if self.s0 == 'd':
            s0 = d
        elif self.s0 == '4d':
            s0 = 4*d
        else:
            raise ValueError("Forbidden s0 value")

        run_logs = []
        for k in range(self.num_tests):
            print(f"Iteration {k+1}/{self.num_tests}")
            print("Generating data...", end=" ", flush=True)
            X, adj = generate(d, s0, N, noise_type=self.data_type, GP=True)
            print("Done")

            start = time.time()
            A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, pruning=self.pruning, threshold=threshold)
            print(f"SCORE execution time: {round(time.time() - start, 2)}s")

            start = time.time()
            fn, fp, rev, SHD, SID, top_order_errors = self.metrics(A_SCORE, adj, top_order_SCORE, compute_SID)
            print(f"Metrics: SHD - {SHD}, SID - {SID}. Computation time: {round(time.time()-start, 2)}s")
            run_logs.append([d, s0, N, threshold, fn, fp, rev, SHD, SID, top_order_errors, SCORE_time, tot_time])

        self.config_logs(run_logs, compute_SID)

    def run_experiment(self, N, eta_G, eta_H):
        start = time.time()
        param_grid = list(ParameterGrid({'d': self.d_values, 'threshold': self.thresholds}))
        run = 0
        tot_runs = len(param_grid)
        for params in param_grid:
            print(f"Start experiment {run+1}/{tot_runs} with parameters: {params}")
            self.run_config(params, N, eta_G, eta_H)
            self.save_logs()

            run += 1
            print(f"Run {run}/{tot_runs} ------ {round(time.time() - start, 2)}s ------ \n")


if __name__ == "__main__":
    """
    Run both s0=d and s0=4*d experiments. Logs in different files
    """
    ########## COMMON PARAMETERS ##########
    # Regression
    N = 1000
    eta_G = 0.001
    eta_H = 0.001

    # General
    data_type = 'Gauss'
    num_tests = 10

    # Pruning algorithm: ["Fast", "FastCAM", "CAM"]
    pruning = "FastCAM"

    if pruning == "Fast" or pruning == "FastCAM":
        # d_values = [10, 20, 50, 100, 200, 500, 1000]
        # thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
        # for s0 in ['d', '4d']:
        d_values = [10, 20, 50, 100, 200]
        thresholds = [0.01]
        for s0 in ['d', '4d']:
            # output_file = f"{pruning.lower()}_{s0}_{data_type}_{d_values[-1]}.csv"
            output_file = f"{pruning.lower()}_{s0}_{data_type}_{d_values[-1]}.csv"
            experiment = FastExperiment(d_values, num_tests, s0, data_type, thresholds, pruning, output_file)
            experiment.run_experiment(N, eta_G, eta_H)

    elif pruning == "CAM":
        d_values = [10, 20, 50, 100]
        cam_cutoff=0.001

        for s0 in ['d', '4d']:
            output_file = f"cam_{s0}_{data_type}_{d_values[-1]}.csv"
            experiment = CAMExperiment(d_values, num_tests, s0, data_type, cam_cutoff, output_file)
            experiment.run_experiment(N, eta_G, eta_H)

    else:
        raise ValueError("Unknown pruning method")