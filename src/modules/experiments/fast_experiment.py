import numpy as np
import time

from sklearn.model_selection import ParameterGrid

from modules.utils import generate, pretty_evaluate
from modules.stein import SCORE, cam_pruning, fast_pruning
from modules.experiments.experiment import Experiment

class FastExperiment(Experiment):
    def __init__(self, d_values, num_tests, s0, data_type, cam_cutoff, output_file, thresholds, pruning):
        super.__init__(d_values, num_tests, s0, data_type, cam_cutoff, output_file)
        self.thresholds = thresholds
        self.pruning = pruning

        self.logs = []
        self.columns = [
            'V', 'E', 'N', 'threshold', 'fn', 'fp', 'reversed', 'SHD', 'SID' , 'D_top', 'SCORE time [s]','Total time [s]'
        ]

    def get_params(self):
        return list(ParameterGrid({'d': self.d_values, 'threshold': self.thresholds}))

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


    def fast(self, X, adj, eta_G, eta_H, threshold, d, s0, N, compute_SID, run_logs):
        """
        Run SCORE with Fast as pruning algorithm. Update logs
        """
        A_SCORE, top_order_SCORE, SCORE_time, tot_time =  SCORE(X, eta_G, eta_H, pruning="Fast", threshold=threshold)
        fn, fp, rev, SHD, SID, top_order_errors = self.metrics(A_SCORE, adj, top_order_SCORE, compute_SID)
        pretty_evaluate(self.pruning, threshold, adj, A_SCORE, top_order_errors, SCORE_time, tot_time, compute_SID)
        run_logs.append([d, s0, N, threshold, fn, fp, rev, SHD, SID, top_order_errors, SCORE_time, tot_time])

        return A_SCORE, top_order_SCORE, SCORE_time, tot_time


    def run_config(self, params, N, eta_G, eta_H):
        d = params['d']
        threshold = params['threshold']
        s0 = self.set_s0(d)

        compute_SID = True
        if d > 200:
            compute_SID = False

        run_logs = []
        for k in range(self.num_tests):
            print(f"Iteration {k+1}/{self.num_tests}")
            X, adj = generate(d, s0, N, noise_type=self.data_type, GP=True)
            self.fast(X, adj, eta_G, eta_H, threshold, d, s0, N, compute_SID, run_logs)

        self.config_logs(run_logs, compute_SID)
        self.save_logs()
