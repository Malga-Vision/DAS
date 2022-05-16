import numpy as np
import pandas as pd
import time

from sklearn.model_selection import ParameterGrid
from modules.utils import generate, pretty_evaluate
from modules.stein import cam_pruning
from modules.experiments.fast_experiment import FastExperiment

class FastCAMExperiment(FastExperiment):
    def __init__(self, d_values, num_tests, s0, data_type, cam_cutoff, thresholds, k):
        super().__init__(d_values, num_tests, s0, data_type, thresholds, k)
        
        self.cam_cutoff = cam_cutoff
        if k is None:
            self.fast_output = f"../logs/exp/fast_{s0}_{d_values[-1]}.csv"
            self.fastcam_output = f"../logs/exp/fastcam_{s0}_{d_values[-1]}.csv"
        else:
            self.fast_output = f"../logs/exp/kfast_{k}_{s0}_{d_values[-1]}.csv"
            self.fastcam_output = f"../logs/exp/kfastcam_{k}_{s0}_{d_values[-1]}.csv"

        self.fast_logs = []
        self.fastcam_logs = []

    def get_params(self):
        return list(ParameterGrid({'d': self.d_values, 'threshold': self.thresholds, 'k': [self.k]}))

    def save_logs(self, logtype):
        if logtype=="fast":
            df = pd.DataFrame(self.fast_logs, columns=self.columns)
            df.to_csv(self.fast_output)
        else:
            df = pd.DataFrame(self.fastcam_logs, columns=self.columns)
            df.to_csv(self.fastcam_output)
            

    def config_logs(self, run_logs, sid, logtype):
        mean_logs = np.mean(run_logs, axis=0)
        std_logs = np.std(run_logs, axis=0)
        logs = []
        for i in range(len(self.columns)):
            m = mean_logs[i]
            s = std_logs[i]
            if self.columns[i] in ["V", "E", "N"]:
                logs.append(f"{int(m)}")
            elif self.columns[i] == 'threshold':
                logs.append(round(m, 5))
            elif not sid and self.columns[i] == "SID":
                logs.append(None)
            else:
                logs.append(f"{round(m, 2)} +- {round(s, 2)}")
        
        if logtype == "fast":
            self.fast_logs.append(logs)
        else:
            self.fastcam_logs.append(logs)


    def fastcam(self, X, adj, threshold, d, s0, N, A_SCORE, top_order_SCORE, SCORE_time, fast_time, sid, run_logs):
        """
        Apply CAM pruning to adjacency matrix found by Fast pruning. Update logs
        """
        start = time.time()
        A_SCORE = cam_pruning(A_SCORE, X, self.cam_cutoff)
        tot_time = SCORE_time + (time.time() - start)

        fn, fp, rev, SHD, SID, top_order_errors = self.metrics(A_SCORE, adj, top_order_SCORE, sid)
        print(pretty_evaluate("FastCAM", threshold, adj, A_SCORE, top_order_errors, SCORE_time, tot_time, sid=sid, s0=s0, K=self.k))
        run_logs.append([d, s0, N, threshold, fn, fp, rev, SHD, SID, top_order_errors, SCORE_time, tot_time])


    def run_config(self, params, N, eta_G, eta_H):
        d = params['d']
        threshold = params['threshold']
        s0 = self.set_s0(d)
        sid = self.compute_SID(d)

        fast_logs = []
        fastcam_logs = []
        for k in range(self.num_tests):
            print(f"Iteration {k+1}/{self.num_tests}")
            X, adj = generate(d, s0, N, noise_type=self.data_type, GP=True)
            
            A_SCORE, top_order_SCORE, SCORE_time = self.fast(X, adj, eta_G, eta_H, threshold, d, s0, N, sid, fast_logs)
            self.fastcam(X, adj, threshold, d, s0, N, A_SCORE, top_order_SCORE, SCORE_time, -1, sid, fastcam_logs)


        self.config_logs(fast_logs, sid, "fast")
        self.save_logs("fast")

        self.config_logs(fastcam_logs, sid, "fastcam")
        self.save_logs("fastcam")
