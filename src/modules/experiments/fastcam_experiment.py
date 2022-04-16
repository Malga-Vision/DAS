import numpy as np
import pandas as pd
import time

from sklearn.model_selection import ParameterGrid
from torch import overrides

from modules.utils import generate, pretty_evaluate
from modules.stein import SCORE, cam_pruning, fast_pruning
from modules.experiments.fast_experiment import FastExperiment

class FastCAMExperiment(FastExperiment):
    def __init__(self, d_values, num_tests, s0, data_type, cam_cutoff, thresholds):
        super().__init__(d_values, num_tests, s0, data_type, thresholds)
        
        self.cam_cutoff = cam_cutoff
        self.fast_output = f"../logs/exp/fast_{s0}_median_{d_values[-1]}.csv"
        self.fastcam_output = f"../logs/exp/fastcam_{s0}_median_{d_values[-1]}.csv"
        self.fast_logs = []
        self.fastcam_logs = []

    def get_params(self):
        return list(ParameterGrid({'d': self.d_values, 'threshold': self.thresholds}))

    def save_logs(self, logs, path):
        df = pd.DataFrame(logs, columns=self.columns)
        df.to_csv(path)

    def config_logs(self, run_logs, compute_SID):
        logs = []
        mean_logs = np.mean(run_logs, axis=0)
        std_logs = np.std(run_logs, axis=0)
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
        
        return logs


    def fastcam(self, X, adj, threshold, d, s0, N, A_SCORE, top_order_SCORE, SCORE_time, fast_time, compute_SID, run_logs):
        """
        Apply CAM pruning to adjacency matrix found by Fast pruning. Update logs
        """
        start = time.time()
        A_SCORE, cam_pruning(A_SCORE, X, self.cam_cutoff)
        tot_time = fast_time + (time.time() - start)

        fn, fp, rev, SHD, SID, top_order_errors = self.metrics(A_SCORE, adj, top_order_SCORE, compute_SID)
        pretty_evaluate("FastCAM", threshold, adj, A_SCORE, top_order_errors, SCORE_time, tot_time, compute_SID)
        run_logs.append([d, s0, N, threshold, fn, fp, rev, SHD, SID, top_order_errors, SCORE_time, tot_time])


    def run_config(self, params, N, eta_G, eta_H):
        d = params['d']
        threshold = params['threshold']
        s0 = self.set_s0(d)

        compute_SID = True
        if d > 200:
            compute_SID = False

        fast_logs = []
        fastcam_logs = []
        for k in range(self.num_tests):
            print(f"Iteration {k+1}/{self.num_tests}")
            X, adj = generate(d, s0, N, noise_type=self.data_type, GP=True)
            
            A_SCORE, top_order_SCORE, SCORE_time, tot_time = self.fast(X, adj, eta_G, eta_H, threshold, d, s0, N, compute_SID, fast_logs)
            self.fastcam(X, adj, threshold, d, s0, N, A_SCORE, top_order_SCORE, SCORE_time, tot_time, compute_SID, fastcam_logs)


        self.fast_logs = self.config_logs(fast_logs, compute_SID)
        self.save_logs(self.fast_logs, self.fast_output)

        self.fastcam_logs = self.config_logs(fastcam_logs, compute_SID)
        self.save_logs(self.fastcam_logs, self.fastcam_output)
