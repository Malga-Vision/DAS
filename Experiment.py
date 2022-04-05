import abc
import time
import cdt
import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid

from utils import *
from stein import *

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
    def config_logs(self):
        """Logs of a tested configuration"""
        raise NotImplementedError

    def metrics(self, A_SCORE, adj, top_order_SCORE):
            fn, fp, rev = edge_errors(A_SCORE, adj)
            SHD = sum((fn, fp, rev))
            SID = int(cdt.metrics.SID(target=adj, pred=A_SCORE))
            top_order_errors = num_errors(top_order_SCORE, adj)
            return fn, fp, rev, SHD, SID, top_order_errors

    def save_logs(self):
        df = pd.DataFrame(self.logs, columns =self.columns)
        df.to_csv(self.output_file)



class CAMExperiment(Experiment):
    def __init__(self, d_values, num_tests, s0, cam_cutoff, output_file):
        self.d_values = d_values
        self.num_tests = num_test
        self.s0 = s0
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
        d = params[0] # Fix
        if self.s0 == 'd':
            s0 = d
        elif self.s0 == '4d':
            s0 = 4*d
        else:
            raise ValueError("Forbidden s0 value")
        
        run_logs = []
        for k in range(self.num_tests):
            X, adj = generate(d, s0, N, GP=True)

            A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, self.cam_cutoff, pruning="CAM")
            fn, fp, rev, SHD, SID, top_order_errors = self.metrics(A_SCORE, adj, top_order_SCORE)
            run_logs.append([d, s0, N, fn, fp, rev, SHD, SID, top_order_errors, SCORE_time, tot_time])

        self.config_logs(run_logs)

    
    def run_experiment(N, eta_G, eta_H):
        start = time.time()
        run = 0
        tot_runs = len(self.d_values)
        for d in self.d_values:
            self.run_config(d, N, eta_G, eta_H)
            self..save_logs()

            run += 1
            print(f"Run {run}/{tot_runs} ------ {round(time.time() - start, 2)}s ------")


class SteinFastExperiment(Experiment):
    def __init__(self, d_values, num_tests, s0, thresholds, output_file):
        self.d_values = d_values
        self.num_tests = num_test
        self.s0 = s0
        self.thresholds = thresholds

        self.logs = []
        self.columns = [
            'V', 'E', 'N', 'threshold', 'fn', 'fp', 'reversed', 'SHD', 'SID' , 'D_top', 'SCORE time [s]','Total time [s]'
        ]
        self.output_file = output_file

    def run_config(self, d, N, eta_G, eta_H):
        pass



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
